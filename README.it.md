# ZeroAI

**Ottimizzazione evolutiva del model merging: cerca automaticamente i pesi di fusione per layer di un merge di modelli LLM.**

**Lingua**: Italiano (questo file) · [English](README.md)

ZeroAI riceve in input un *base model* e N *expert model* (checkpoint `.safetensors`) e cerca, tramite Optuna (TPE) o un algoritmo genetico, la combinazione migliore di:

- **metodo di merge** (`linear`, `taskslim`, `ties`, `dare_linear`, `dare_ties`),
- **coefficienti per-model × per-blocco** (uno slot di peso per ogni gruppo di layer),
- **intensità di pruning** (trim di magnitudine e drop DARE),
- **rescaling** e **normalizzazione** dei coefficienti,

misurando ogni candidato con una batteria di benchmark multi-dominio. Il risultato è il **genoma** ottimo (la configurazione migliore trovata), esportato come `best_genome.json` + `best_mergekit.yaml` pronto per mergekit, più il report completo di tutti i candidati provati.

> **Nota di onestà.** La modalità `mock` (mondo sintetico) e la modalità `model` lato *merge tensoriale* sono pienamente funzionanti e coperte dai test. La valutazione su LLM reali richiede checkpoint reali e VRAM: è l'unico anello che dipende dall'ambiente, non dal codice. Vedi [Limiti noti](#15-limiti-noti-e-troubleshooting).

---

## Indice

1. [Che problema risolve](#1-che-problema-risolve)
2. [Quick start (60 secondi)](#2-quick-start-60-secondi)
3. [Architettura](#3-architettura)
4. [Il genoma: cosa viene ottimizzato](#4-il-genoma-cosa-viene-ottimizzato)
5. [Metodi di merge implementati](#5-metodi-di-merge-implementati)
6. [Blocchi: i coefficienti per layer](#6-blocchi-i-coefficienti-per-layer)
7. [Backend di merge: native e mergekit](#7-backend-di-merge-native-e-mergekit)
8. [Valutazione: mock e model](#8-valutazione-mock-e-model)
9. [Ottimizzatori: Optuna e algoritmo genetico](#9-ottimizzatori-optuna-e-algoritmo-genetico)
10. [La pipeline e la gestione degli errori](#10-la-pipeline-e-la-gestione-degli-errori)
11. [Artefatti prodotti da ogni run](#11-artefatti-prodotti-da-ogni-run)
12. [Installazione](#12-installazione)
13. [Riferimento completo della CLI](#13-riferimento-completo-della-cli)
14. [Esempi pratici](#14-esempi-pratici)
15. [Limiti noti e troubleshooting](#15-limiti-noti-e-troubleshooting)
16. [Uso come libreria Python](#16-uso-come-libreria-python)
17. [Test](#17-test)
18. [Struttura del repository](#18-struttura-del-repository)

---

## 1. Che problema risolve

Il model merging fonde più fine-tune dello stesso base model senza riaddestrare nulla, lavorando nello **spazio dei task vector**:

```
delta_task   = pesi(expert) - pesi(base)
pesi(merged) = base + somma_m ( w_m * delta_m )
```

Il problema pratico è che **un singolo peso globale non basta**: un expert di codice eccelle nei layer medio-bassi, un expert di ragionamento nei layer profondi, e fondere tutto con peso 1.0 produce **interferenza distruttiva** (i delta si annullano a vicenda) e un'esplosione della norma del task vector fuso, che degrada la capacità generale del modello.

ZeroAI trasforma queste scelte in un problema di ottimizzazione:

| Scelta normalmente manuale | In ZeroAI è un gene del genoma |
|---|---|
| Quale algoritmo di merge usare | `method` |
| Quanto pesa ogni expert | `weights[model][block]` |
| Quanto pruning applicare | `density`, `dare_density` |
| Quanto rescalare i delta sopravvissuti | `dare_scale` |
| Se normalizzare i coefficienti | `normalize` |

Le tecniche implementate (magnitude trimming, elect-sign, DARE drop+rescale) rispecchiano la semantica di mergekit 0.1.x e i paper **TIES-Merging** e **DARE**.

---

## 2. Quick start (60 secondi)

```powershell
# 1. verifica l'ambiente (python, torch, CUDA, mergekit)
python -m zeroai.cli doctor

# 2. giro completo in modalità mock (nessun modello richiesto, ~1 secondo)
python -m zeroai.cli run --mode mock --optimizer genetic --population 6 --generations 3 --blocks 3 --run-name smoke_ga

# 3. stesso giro con Optuna
python -m zeroai.cli run --mode mock --optimizer optuna --trials 12 --blocks 3 --run-name smoke_optuna

# 4. genera SOLO il YAML mergekit di una configurazione manuale
python -m zeroai.cli yaml --mode mock --models a b --method dare_ties --weights 0.2 0.8 --blocks 2

# 5. esegui UN solo merge (nessuna ottimizzazione) e scrivi il checkpoint
python -m zeroai.cli merge --mode mock --models a b --method dare_ties --density 0.6 --blocks 2 --write-checkpoints
```

Ogni `run` termina con l'elenco degli artefatti scritti:

```
artifacts:
  best_genome:     runs\smoke_ga\best_genome.json
  mergekit_yaml:   runs\smoke_ga\best_mergekit.yaml
  candidates_csv:  runs\smoke_ga\candidates.csv
  summary:         runs\smoke_ga\summary.md
  summary_json:    runs\smoke_ga\summary.json
  run_dir:         runs\smoke_ga
```

`python -m zeroai.cli ...` e `python -m zeroai ...` sono equivalenti. Dopo `pip install -e .` è disponibile anche il comando `zeroai`.

---

## 3. Architettura

Il flusso è un **loop chiuso**: l'ottimizzatore propone un genoma, la pipeline lo esegue e lo valuta, il punteggio torna all'ottimizzatore.

```
                    +--------------------------------------------------------------+
                    |            Optimizer (Optuna TPE / algoritmo genetico)      |
                    |   propone Genome = metodo + pesi per blocco + pruning        |
                    +-------------------------------+-----------------------------+
                                                    | genome
                                                    v
   N modelli  -->  CandidateGenerator  -->  MergePlan + mergekit YAML
   + base             |   (normalizzazione per blocco, validazione)
                      v
                MergeBackend  -->  MergedArtifact
     (native torch | mergekit subprocess)   (checkpoint + delta uniti)
                      |
                      v
                 Evaluator  -->  EvaluationReport (fitness + metriche)
        (proxy sintetico | benchmark LLM reale)
                      |
                      +---- fitness ----> ripetere fino a budget esaurito
                                          |
                                          v
                                    write_report()
```

| Modulo | File | Responsabilità |
|---|---|---|
| Contratti dati | `zeroai/schemas.py` | `MergeMethod`, `DType`, `ModelSpec`, `BlockSpec`, `MergePlan`, `MergedArtifact`, `EvaluationReport`, `SearchSpace` |
| Cromosoma | `zeroai/genome.py` | `Genome` immutabile: `sample` / `uniform` / `random`, `clipped`, `crossover`, `mutate`, `key()`, (de)serializzazione, `normalize_weights` |
| Partizione layer | `zeroai/blocks.py` | `partition_tensors`, `layer_index`, `build_block_index` |
| Modulo 1 - candidato | `zeroai/candidate.py` | `CandidateGenerator`: Genome -> `MergePlan` -> YAML mergekit |
| Modulo 2 - merge | `zeroai/merge/` | `algorithms.py` (trim, elect-sign, DARE), `backend.py` (motore nativo), `mergekit_backend.py` + `mergekit_driver.py`, `tensor_store.py` |
| Modulo 3 - valutazione | `zeroai/eval/` | `base.py` (interfaccia), `synthetic.py` (proxy sintetico), `benchmarks.py` (suite di test), `llm.py` (inferenza reale) |
| Modulo 4 - ottimizzazione | `zeroai/optimizers/` | `base.py`, `optuna_opt.py`, `genetic.py` |
| Modulo 5 - orchestrazione | `zeroai/pipelines.py` | `PipelineConfig`, `EvolutionaryMergePipeline`, `PipelineResult` |
| Modulo 6 - report | `zeroai/report.py` | `write_report`, `render_summary` |
| Interfaccia | `zeroai/cli.py`, `zeroai/__main__.py` | `run`, `yaml`, `merge`, `doctor` |

I moduli comunicano **solo** tramite i contratti di `schemas.py`: backend, evaluator e optimizer sono intercambiabili e testabili in isolamento.

---

## 4. Il genoma: cosa viene ottimizzato

`zeroai/genome.py` definisce `Genome`, una dataclass **frozen** (immutabile): può essere usata come chiave di dizionario, inserita in un set per il dedup, passata a Optuna o messa nella popolazione del GA senza copie difensive.

| Gene | Tipo | Significato | Range di default in `SearchSpace` |
|---|---|---|---|
| `method` | `MergeMethod` | strategia di fusione | tutti e 5 i metodi |
| `weights` | tuple `[model][block]` | coefficiente di ogni expert per ogni blocco di layer | `0.0 .. 1.5` |
| `density` | float | **frazione di coefficienti mantenuti** dal trim di magnitudine (usata da `taskslim`, `ties`, `dare_ties`) | `0.05 .. 1.0` |
| `dare_density` | float | **frazione di coefficienti scartati** con drop bernoulliano DARE | `0.0 .. 0.9` |
| `dare_scale` | float | fattore di rescaling aggiuntivo sui delta sopravvissuti | `1.0 .. 3.0` |
| `normalize` | bool | se ricondurre a somma 1 i coefficienti di ogni blocco | `True` |
| `dtype` | `DType` | dtype di scrittura del checkpoint | `float16` |

Metodi principali di `Genome`:

- `key()` - stringa identità stabile (metodo + parametri scalari + pesi arrotondati a 6 decimali) usata per **deduplicare** le valutazioni;
- `sample()` / `Genome.random()` - disegno uniforme dentro lo `SearchSpace`;
- `Genome.uniform(n_models, n_blocks)` - tutti gli expert con peso `1/n_models` su ogni blocco (il merge "naif");
- `clipped(space)` - riporta ogni gene dentro i limiti (usato dopo crossover e mutazione);
- `crossover(other, rng, alpha=0.5)` - **blend crossover BLX-alpha** sui geni continui + crossover uniforme (50/50) sui geni discreti (`method`, `normalize`);
- `mutate(rng, rate, sigma)` - mutazione gaussiana (sigma) sui continui e ri-disegno del gene `method`;
- `to_dict()` / `from_dict()` - serializzazione JSON: è il formato di `best_genome.json` e di `--genome`;
- `short_repr()` - riga leggibile usata nei log.

> **Attenzione a `density` vs `dare_density`.** `density` è la frazione **mantenuta**, `dare_density` è la frazione **scartata**. Nel YAML mergekit entrambi finiscono nella chiave `density` di mergekit (che è sempre frazione mantenuta), quindi per i metodi DARE ZeroAI scrive `1 - dare_density`.

### Normalizzazione dei pesi

`normalize_weights()` in `zeroai/genome.py` normalizza **per colonna, cioè per blocco** (non per modello come fa mergekit): la somma dei valori assoluti dei coefficienti dei modelli dentro ogni blocco viene portata a 1. Questo tiene sotto controllo la norma del task vector fuso blocco per blocco, che è la causa principale del collasso della capacità generale. Se vuoi pesi "pass-through" senza riscalatura, usa `--no-normalize`.

---

## 5. Metodi di merge implementati

Tutti in `zeroai/merge/algorithms.py`. L'accumulo aritmetico avviene sempre in **float32** (`WORK_DTYPE`) indipendentemente dal dtype di storage, come fa mergekit internamente: è ciò che rende stabili i merge in fp16.

| Metodo | Pipeline per tensor | Geni usati |
|---|---|---|
| `linear` | somma pesata dei delta | `weights` |
| `taskslim` | trim di magnitudine (mantiene `density`) -> somma pesata | `weights`, `density` |
| `ties` | trim -> **elect-sign** (segno di maggioranza pesata) -> scarta i coefficienti in disaccordo col segno -> somma pesata | `weights`, `density` |
| `dare_linear` | **DARE drop** bernoulliano (mantiene `1 - dare_density`, rescale L2) x `dare_scale` -> somma pesata | `weights`, `dare_density`, `dare_scale` |
| `dare_ties` | DARE drop+rescale -> trim -> elect-sign -> scarta gli outlier di segno -> somma pesata | tutti |

Dettagli implementativi:

- `magnitude_trim(tensor, density)` - mantiene i `density * numel` coefficienti con magnitudine maggiore (`topk` su `abs`).
- `bernoulli_drop(tensor, density)` - maschera bernoulliana; su CPU con fp16 upcasta a float32 perché `torch.bernoulli` non supporta fp16 su CPU.
- `_rescaled_masked_tensor` - dopo la maschera ripristina la norma originale (`l1`, `l2`, `linf`) esattamente come `rescale_norm` di mergekit (il passo "REscale" del paper DARE). Per il DARE ZeroAI usa `l2`; per il trim usa `None` (nessun rescale).
- `elect_sign(tensors, weights)` - accumula `weight * tensor` e ne prende il segno; i casi di somma zero sono risolti verso `+1` per non scartare silenziosamente un coefficiente.
- `interference_energy(contributions)` - restituisce `(cross_energy, self_energy)`: `cross_energy` è la somma dei prodotti scalari incrociati fra contributi di modelli diversi. Valori negativi = interferenza distruttiva. Finisce nelle `stats` di ogni candidato e quindi in `candidates.csv` / `summary.json`.

Ogni metodo ha un nome equivalente nel YAML mergekit (`MergeMethod.mergekit_method`): i nomi sono identici (`linear`, `taskslim`, `ties`, `dare_linear`, `dare_ties`).

---

## 6. Blocchi: i coefficienti per layer

Un **blocco** è un gruppo contiguo di layer che riceve un proprio coefficiente di fusione: è la struttura che trasforma "un peso globale per modello" in "un peso per modello per zona di profondità".

`zeroai/blocks.py::partition_tensors(names, num_blocks)`:

1. deduplica e ordina i nomi dei tensor;
2. estrae l'indice di layer con tre regex: `...layers.<n>.`, `...h.<n>.` / `block.<n>.` / `decoder_layer.<n>.`, e il pattern interno `block_<n>.` usato dal mondo sintetico;
3. distribuisce i layer sui blocchi in modo proporzionale alla profondità: `block = floor((idx + 1) / (max_idx + 1) * num_blocks)`;
4. **pina le embedding nel primo blocco** (`embed_tokens`, `word_embeddings`, `wte`, `embeddings.`, `tok_embeddings`) e **le teste di output nell'ultimo** (`lm_head`, `output.`, `final_dense`, `classifier`);
5. i nomi di cui non si può dedurre la profondità (checkpoint esotici) vengono distribuiti uniformemente fra i blocchi invece di accumularsi nel blocco 0.

`--blocks N` controlla quanti slot di coefficiente per modello: con 2 modelli e `--blocks 4` il genoma ha 8 pesi liberi. Con `--blocks 1` il problema degenera nel classico merge lineare a peso unico.

`MergePlan.validate()` verifica che la matrice dei pesi abbia esattamente `n_models` righe e `n_blocks` colonne, che `0 < density <= 1`, `0 <= dare_density < 1` e `dare_scale > 0`. Il backend nativo verifica inoltre che `plan.blocks` copra **tutti** i tensor comuni fra base ed expert.

---

## 7. Backend di merge: native e mergekit

### `NativeMergeBackend` (default, sempre disponibile)

`zeroai/merge/backend.py`. Motore torch in-process:

- legge i tensor dai `TensorStore` (in-memory per la modalità mock, `SafetensorsTensorStore` per i checkpoint reali);
- considera solo i **nomi di tensor comuni** fra base ed expert (`common_tensor_names`);
- calcola i delta `expert - base`, applica `combine(...)` a ogni tensor con il coefficiente del blocco corrispondente;
- produce `base + delta_fuso` scritto nel `dtype` richiesto;
- accumula le `stats` di interferenza (`interference_cross_energy`, `interference_self_energy`, `interference_ratio`, `merged_delta_norm`, `base_norm`);
- con `save_output=True` scrive `<out>/model.safetensors` (metadato `format: pt`) e **copia tutti i file non-`.safetensors` della directory del base model** (`config.json`, tokenizer, chat template, ecc.) nella directory del candidato, così il candidato è caricabile da `transformers`.

`SafetensorsTensorStore` carica in modo **lazy e con cache**: solo i tensor effettivamente toccati vengono materializzati, e restano in cache per tutta la loop evolutiva.

### `MergekitBackend` (out-of-process)

`zeroai/merge/mergekit_backend.py` + `zeroai/merge/mergekit_driver.py`.

mergekit 0.1.x richiede `huggingface_hub>=1.31`, mentre transformers 4.x richiede `huggingface_hub<1.0`: installarli nello stesso ambiente rompe transformers. Per questo ZeroAI **non importa mergekit nel proprio interprete**:

- il **YAML è sempre generato** dal `CandidateGenerator` (ed è testabile senza mergekit installato);
- l'esecuzione avviene lanciando `python -m zeroai.merge.mergekit_driver --config <yaml> --out <dir> --device <dev>` con l'interprete indicato da `--mergekit-python`, cioè un **venv dedicato a mergekit**;
- il driver importa `mergekit.merge.run_merge` con `copy_tokenizer=True` e restituisce un JSON con i tempi; codici di uscita: `3` = mergekit non disponibile, `4` = configurazione non valida.

`--merge-backend`:

| Valore | Comportamento |
|---|---|
| `native` | motore torch interno (**default della CLI**) |
| `mergekit` | sempre driver esterno; fallisce se mergekit non è raggiungibile |
| `auto` | prova mergekit in-process, altrimenti ripiega su native (default di `PipelineConfig` quando ZeroAI è usato come libreria; in modalità `mock` usa comunque native) |

`MergekitBackend.is_available()` (usata da `doctor`) verifica `python -c "import mergekit.merge"` nell'interprete indicato.

### YAML generato

```yaml
# Generated by ZeroAI CandidateGenerator
# method=dare_ties blocks=2 density(keep)=0.6000 dare_drop=0.4000 dare_scale=1.2000
models:
- model: synthetic://e1
  parameters:
    weight:
    - 0.2
    - 0.8
    density: 0.6
    rescale_norm: l2
    normalize: true
- model: synthetic://e2
  parameters:
    weight:
    - 0.8
    - 0.2
    density: 0.6
    rescale_norm: l2
    normalize: true
merge_method: dare_ties
base_model: synthetic://base
parameters:
  density: 0.6
  normalize: true
dtype: float16
```

È l'output reale di `python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base --method dare_ties --weights 0.2 0.8 0.8 0.2 --density 0.6 --dare-density 0.4 --dare-scale 1.2 --blocks 2`.

Come leggerlo:

- `weight` è una **lista di pesi interpolati** di mergekit: un valore per blocco, interpolato linearmente da mergekit sulla posizione normalizzata del layer. È il meccanismo nativo di mergekit per i pesi layer-wise.
- In `--mode mock` i path emessi sono `synthetic://<label>`; in `--mode model` sono esattamente i path che passi a `--models` / `--base`.
- `density` dentro i `parameters` di ogni modello è sempre la frazione **mantenuta**: per i metodi DARE ZeroAI scrive `1 - dare_density` (qui 0.4 scartato -> 0.6 mantenuto). `rescale_norm: l2` compare solo nei metodi DARE, `normalize: true` solo quando il gene `normalize` è attivo.
- Il blocco `parameters` globale è il default di mergekit per i modelli che non dichiarano quelle chiavi.

> Esempio reale di normalizzazione: con `--weights 0.2 0.8` e normalizzazione attiva (default) la colonna del blocco viene riscalata in `[0.5, 0.5]`. Se vuoi che 0.2/0.8 restino tali, passa `--no-normalize`.

---

## 8. Valutazione: mock e model

La modalità scelta con `--mode` determina **come viene misurato un candidato**.

### 8.1 `--mode mock` (default): proxy sintetico deterministico

`zeroai/eval/synthetic.py::build_synthetic_world()` costruisce un "mondo di merge" riproducibile (stesso `--seed` -> stesse matrici):

- un base model di tensor piccoli (`synthetic_tensor_shape` di default `32x32`, `synthetic_tensors_per_block=2`);
- un vettore "verità" per dominio (`reasoning` peso 0.30, `code` 0.25, `math` 0.20, `general` 0.25), costruito come **profilo di profondità** (gaussiana centrata su `t=0.65` per reasoning, `0.35` per code, `0.5` per math/general) più una **core condivisa** fra domini, per modellare capacità non ortogonali nello spazio dei pesi;
- N expert la cui copertura dei domini ruota (ogni expert è forte su un sottoinsieme di domini);
- **conflitti di segno voluti** su una frazione di coppie (expert, blocco): esattamente il caso che l'elect-sign deve risolvere;
- rumore gaussiano indipendente per expert (`synthetic_noise=0.35`).

Il punteggio non richiede inferenza: valuta il **vettore task fuso** (`artifact.deltas`) prodotto dal backend nativo.

| Metrica | Come viene calcolata |
|---|---|
| `capability` | cosine similarity fra il vettore fuso di ogni blocco e il vettore truth del dominio, **ponderata sul profilo di profondità** del dominio, mediata sui domini |
| `purity` | frazione dell'energia del delta fuso che resta **dentro il sottospazio delle capacità**: proiezione least-squares del vettore fuso sui vettori truth di tutti i domini; `1 - energia_residua / energia_totale` |
| `stability` | `1 - abs(norma(delta fuso) - ideal_norm) / ideal_norm`, dove `ideal_norm` è la norma del delta "verità" ponderato sui domini: penalizza l'over-amplificazione che distrugge la capacità generale |
| `cooperation` | `1 - interferenza distruttiva` fra i contributi pesati degli expert (energia incrociata sui delta originali degli expert) |
| `domain_<nome>` | punteggio del singolo dominio (`reasoning`, `code`, `math`, `general`) |

**Fitness** = `0.55 * capability + 0.15 * purity + 0.15 * stability + 0.15 * cooperation` (pesi in `SyntheticBenchmark.fitness_weights`).

Serve a: validare l'architettura, testare gli ottimizzatori, verificare che i coefficienti per blocco contino davvero, e sviluppare senza GPU né download.

### 8.2 `--mode model`: benchmark reale su checkpoint fusi

`zeroai/eval/llm.py::LLMEvaluator` carica il checkpoint del candidato e risponde a una batteria di item offline e deterministici (`zeroai/eval/benchmarks.py::default_suite`, 10 item):

| Dominio | Peso | Item | Checker |
|---|---|---|---|
| `reasoning` | 0.35 | 4 item: velocità media, proporzioni, successioni, percentuali | `NumericAnswerChecker` (parsa l'ultimo numero, tolleranza 1e-4) |
| `code` | 0.35 | 3 item: `sum_squares`, `fizzbuzz`, `reverse_words` | `PythonExecChecker`: estrae il code fence, lo unisce alle `assert` e lo esegue in subprocess con timeout 10 s |
| `general` | 0.30 | spiegazione, riscrittura di tono, formattazione a elenco | `InstructionChecker`: banda di lunghezza, marcatori obbligatori, assenza di frasi di rifiuto |

**Fitness** = media ponderata dei punteggi di dominio (1 = tutti gli item del dominio superati). Viene registrata anche la metrica aggiuntiva `benchmark_pass_rate` (media secca su tutti gli item, peso 0).

Backend di inferenza (`_resolve_backend`):

- `vllm` se installato (molto più veloce sul loop many-candidate), altrimenti
- `transformers` (`AutoModelForCausalLM` + `AutoTokenizer`, `device_map="auto"`, generazione greedy perché `temperature=0`).

Il modello resta caricato finché il candidato successivo non cambia path (`_loaded_from`); `close()` libera la VRAM a fine run.

> **Vincolo importante.** `LLMEvaluator` richiede `artifact.output_path`, quindi in `--mode model` la valutazione reale funziona **solo con `--write-checkpoints`**. Senza, ogni candidato viene marcato `eval: LLMEvaluator needs a merged checkpoint on disk` (fitness 0) e la ricerca non può progredire.

---

## 9. Ottimizzatori: Optuna e algoritmo genetico

### Optuna (`--optimizer optuna`, default)

`zeroai/optimizers/optuna_opt.py`. Esplora lo spazio misto categorico/continuo:

- `suggest_categorical("method", ...)`
- `suggest_float("weight_m{model}_b{block}", weight_low, weight_high)` per ogni coppia modello/blocco
- `suggest_float("density")`, `suggest_float("dare_density")`, `suggest_float("dare_scale")`
- `suggest_categorical("normalize", [True, False])`

Configurazione: `TPESampler(seed=seed, n_startup_trials=min(10, n_trials))`, `MedianPruner(n_startup_trials=5)` (disattivabile con `prune=False`), direzione `maximize`, `gc_after_trial=True`, studio in memoria o su `storage` (es. `sqlite:///zeroai.db` con `load_if_exists`) per riprendere una ricerca interrotta. I genomi passati in `initial_genomes` vengono messi in coda con `enqueue_trial` (`_flatten_genome`), quindi il seme fornito viene sempre valutato.

### Algoritmo genetico (`--optimizer genetic`)

`zeroai/optimizers/genetic.py`.

| Parametro | Default della classe | Default CLI | Note |
|---|---|---|---|
| `population_size` | 24 | `--population` 20 | minimo 4 |
| `generations` | 12 | `--generations` 8 | minimo 1 |
| `mutation_rate` | 0.25 | - | probabilità per gene |
| `mutation_sigma` | 0.15 | - | deviazione gaussiana sui pesi |
| `crossover_rate` | 0.7 | - | blend crossover |
| `method_mutation_rate` | 0.15 | - | ri-disegno indipendente del gene `method` |
| `tournament_size` | 3 | - | selezione a torneo |
| `elite_keep` | 2 | - | elitismo |

Meccanismo: popolazione iniziale (seed + random deduplicati per `key()`), poi per ogni generazione elitismo -> torneo -> crossover -> mutazione -> re-draw del metodo -> `clipped()` -> valutazione con cache interna.

Differenza chiave rispetto a Optuna: il GA **muta anche il gene del metodo di merge**, quindi può uscire da un bacino in cui tutta una famiglia di strategie (per esempio tutte le varianti TIES) è bloccata. È anche il fallback quando optuna non è installato.

### `--optimizer none`

Valuta un singolo genoma (quello fornito con `--genome`, o costruito da `--method`/`--weights`, altrimenti uno random) e produce comunque tutti i report. È la modalità usata internamente dai comandi `yaml` e `merge`.

---

## 10. La pipeline e la gestione degli errori

`zeroai/pipelines.py::EvolutionaryMergePipeline` non contiene logica di ottimizzazione: collega i quattro moduli e gestisce caching, persistenza e resilienza.

### `PipelineConfig` (tutti i campi)

| Campo | Default | Note |
|---|---|---|
| `mode` | `"mock"` | `"mock"` o `"model"` |
| `base_model` / `models` | `None` / `()` | obbligatori in modalità `model` (validato in `__post_init__`) |
| `num_blocks` | 4 | >= 1 |
| `dtype` | `DType.FLOAT16` | `float16` / `bfloat16` / `float32` |
| `seed` | 1337 | determinismo di mondo sintetico, DARE e sampler |
| `run_dir` / `run_name` | `"runs"` / `"run"` | la directory effettiva è `run_dir/run_name`, assolutizzata e creata automaticamente |
| `search_space` | `SearchSpace()` | limiti dei geni |
| `write_checkpoints` | `False` | scrive i checkpoint dei candidati |
| `merge_backend` | `"auto"` | `"native"`, `"mergekit"`, `"auto"` |
| `mergekit_python` / `mergekit_device` | `None` / `"cpu"` | interprete e device del driver mergekit |
| `device` | `"cpu"` | device del merge e dell'evaluator |
| `optimizer` | `"optuna"` | `"optuna"`, `"genetic"`, `"none"` |
| `n_trials` / `population_size` / `generations` | 40 / 20 / 8 | budget di ricerca |
| `initial_genomes` | `()` | semi della ricerca |
| `domains` | `DEFAULT_DOMAINS` | domini del mondo sintetico |
| `benchmark_suite` | `None` | suite personalizzata per `LLMEvaluator` |
| `max_new_tokens` | 256 | generazione dell'evaluator reale |
| `synthetic_tensors_per_block` / `synthetic_tensor_shape` / `synthetic_noise` | 2 / `(32, 32)` / 0.35 | parametri del mondo mock |

### `prepare()`

- **mock**: costruisce il mondo sintetico e crea `ModelSpec` fittizi `synthetic://<label>`. Se `--models` è vuoto vengono generati 3 expert (`expert_0..2`); se `--models` è fornito, i **nomi** dei path diventano le label degli expert (i path non vengono letti).
- **model**: apre un `SafetensorsTensorStore` per il base e per ogni expert, calcola i **nomi di tensor comuni** e partiziona i blocchi su quell'insieme. Se nessun tensor è condiviso, la run fallisce subito con un errore chiaro.

### Resilienza per candidato

`_score()` incapsula ogni candidato:

| Guasto | Esito |
|---|---|
| eccezione nel merge | `CandidateRecord(error="merge: ...")`, fitness 0, ricerca continua |
| eccezione nella valutazione | `error="eval: ..."`, fitness 0, ricerca continua |
| metriche NaN / inf | `error="non-finite fitness"`, fitness 0 |

I candidati falliti restano nel report ma sono **esclusi** da `PipelineResult.top()` e dal blocco "Top candidates" di `summary.md`: un candidato rotto non può far "vincere" una configurazione invalida.

### Deduplica

`_score_cache` è indicizzato su `Genome.key()`: se l'ottimizzatore ripropone una configurazione già vista (molto frequente con elitismo e pruner), non vengono rifatti né il merge né la valutazione. Optuna e il GA hanno a loro volta una cache interna analoga.

---

## 11. Artefatti prodotti da ogni run

Ogni run scrive in `runs/<run-name>/`:

| File | Contenuto |
|---|---|
| `best_genome.json` | genoma ottimo serializzato (method, weights, density, dare_density, dare_scale, normalize, dtype), best fitness, durata, numero di candidati e fallimenti, pesi dei domini, dimensione dei blocchi, e la sezione `optimization` con la **history di convergenza** |
| `best_mergekit.yaml` | configurazione mergekit del genoma ottimo, pronta da eseguire |
| `candidates.csv` | una riga per candidato: `candidate_id, method, fitness, density, dare_density, dare_scale, normalize, weight_m{m}_b{b}..., <tutte le metriche>, merge_seconds, eval_seconds, error` |
| `summary.md` | report leggibile: header della run, best genome, tabella dei coefficienti per blocco, top 5 candidati, metriche del candidato migliore, tabella di convergenza |
| `summary.json` | dump completo machine-readable (`PipelineResult.to_dict()`): tutti i candidati con genoma, metriche, `stats` di interferenza, fallimenti, blocchi, domini, ottimizzazione |
| `candidates/cand_0001/`, `cand_0002/`, ... | solo con `--write-checkpoints`: `model.safetensors` del candidato + i file ausiliari copiati dal base model |

Le `stats` di interferenza disponibili in `summary.json` per ogni candidato sono `tensors`, `blocks_used`, `interference_cross_energy`, `interference_self_energy`, `interference_ratio`, `merged_delta_norm`, `base_norm`: servono a capire **perché** un genoma ha vinto (per esempio un rapporto di interferenza più basso a parità di capability).

---

## 12. Installazione

Requisito: **Python >= 3.10** (dichiarato in `pyproject.toml`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # torch, transformers, accelerate, safetensors, PyYAML, numpy, optuna, tqdm
pip install -e .                     # rende disponibile il comando "zeroai"
```

`pyproject.toml` dichiara anche gli extras:

```powershell
pip install -e ".[test]"      # pytest
pip install -e ".[serve]"     # vllm: backend di inferenza veloce per la modalita model
pip install -e ".[mergekit]"  # mergekit, solo se sai di non avere conflitti con transformers
```

### mergekit in un venv separato (consigliato)

mergekit 0.1.x e transformers 4.x hanno vincoli incompatibili su `huggingface_hub`, e mergekit non risulta installabile pulitamente su Python 3.14. ZeroAI funziona comunque: il YAML lo genera da sé e l'esecuzione mergekit passa per un interprete esterno.

```powershell
python -m venv .venv-mergekit
.\.venv-mergekit\Scripts\Activate.ps1
pip install mergekit
deactivate

# poi, dall'ambiente principale:
python -m zeroai.cli run --mode model --base ./base --models ./e1 ./e2 `
  --merge-backend mergekit `
  --mergekit-python .\.venv-mergekit\Scripts\python.exe `
  --mergekit-device cuda --write-checkpoints --run-name via_mergekit
```

### Verifica dell'ambiente

```powershell
python -m zeroai.cli doctor
```

Esempio di output (macchina con GPU CUDA):

```json
{
  "python": "3.14.5",
  "torch": "2.14.1+cu130",
  "transformers": "4.52.4",
  "accelerate": "1.15.0",
  "safetensors": "0.8.0",
  "optuna": "5.0.0",
  "yaml": "6.0.3",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 5070 Ti",
  "vram_gb": 15.89,
  "mergekit_in_process": false
}
```

`mergekit_in_process: false` **non è un errore**: significa solo che il backend mergekit in-process non è disponibile e che va usato `native` oppure un venv dedicato.

---

## 13. Riferimento completo della CLI

```
usage: zeroai [-h] {run,yaml,merge,doctor} ...

  run     run the closed evolutionary merge loop
  yaml    emit the mergekit YAML for a genome
  merge   execute one merge, no optimisation
  doctor  report environment / GPU / mergekit readiness
```

### 13.1 `run` - il loop evolutivo completo

```powershell
python -m zeroai.cli run [flag]
```

**Flag generali**

| Flag | Default | Descrizione |
|---|---|---|
| `--mode {mock,model}` | `mock` | `mock` = mondo sintetico, nessun file richiesto; `model` = checkpoint `.safetensors` reali |
| `--models M1 M2 ...` | vuoto | path degli expert. In `mock` forniscono solo le label (se vuoto, 3 expert sintetici); in `model` sono i checkpoint da fondere |
| `--base PATH` | `None` | base model (obbligatorio in `model`) |
| `--blocks N` | 4 | numero di slot di coefficiente per modello |
| `--dtype {float16,bfloat16,float32}` | `float16` | dtype del checkpoint scritto e dell'inferenza |
| `--device DEVICE` | `cpu` | device di merge e valutatore (`cuda` per i modelli reali) |
| `--seed INT` | 1337 | determinismo |
| `--run-dir DIR` | `runs` | radice delle run |
| `--run-name NAME` | `run` | sottocartella `runs/NAME` |
| `--log-level LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |

**Flag di ricerca**

| Flag | Default | Descrizione |
|---|---|---|
| `--optimizer {optuna,genetic,none}` | `optuna` | motore di ricerca |
| `--trials N` | 40 | budget di Optuna |
| `--population N` | 20 | dimensione popolazione del GA |
| `--generations N` | 8 | generazioni del GA |
| `--methods linear ties ...` | tutti | restringe il gene `method` (accetta i 5 nomi dei metodi) |
| `--weight-low` / `--weight-high` | 0.0 / 1.5 | range dei coefficienti per blocco |
| `--density-low` / `--density-high` | 0.05 / 1.0 | range della frazione mantenuta dal trim |
| `--dare-low` / `--dare-high` | 0.0 / 0.9 | range della frazione scartata da DARE |
| `--genome FILE.json` | `None` | genoma seed letto da file (per esempio un `best_genome.json` di una run precedente) |

> `dare_scale` (range 1.0-3.0) **non ha flag CLI**: si configura solo tramite `SearchSpace` usando ZeroAI come libreria.

**Flag di genoma esplicito** (inutili con un ottimizzatore attivo, fondamentali per `yaml` e `merge`)

| Flag | Default | Descrizione |
|---|---|---|
| `--method NAME` | `None` | metodo di merge fisso |
| `--weights W...` | `None` | coefficienti: `len(blocks)` valori condivisi da tutti i modelli, oppure `blocks*models` valori model-major |
| `--density X` | 1.0 | frazione mantenuta dal trim |
| `--dare-density X` | 0.0 | frazione scartata da DARE |
| `--dare-scale X` | 1.0 | rescaling dei delta sopravvissuti |
| `--normalize` / `--no-normalize` | normalizzazione attiva | ricondurre a 1 la somma dei coefficienti di ogni blocco |

**Flag di backend di merge**

| Flag | Default | Descrizione |
|---|---|---|
| `--merge-backend {auto,native,mergekit}` | `native` | motore di fusione |
| `--mergekit-python PATH` | interprete corrente | python del venv che contiene mergekit |
| `--mergekit-device DEVICE` | `cpu` | device usato dal driver mergekit |
| `--write-checkpoints` | spento | scrive `runs/NAME/candidates/cand_XXXX/model.safetensors` per ogni candidato (**obbligatorio per la valutazione reale**) |

### 13.2 `yaml` - stampa la configurazione mergekit

```powershell
python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base `
  --method ties --weights 0.3 0.7 --blocks 2
```

Accetta gli stessi flag di `run` (comprese le opzioni di ricerca, usate solo per costruire lo `SearchSpace`), forza `optimizer=none` e stampa il YAML su stdout senza eseguire alcun merge. Se non fornisci `--genome` né `--method`/`--weights`, usa `Genome.uniform` (pesi `1/n_models`).

### 13.3 `merge` - un solo merge, nessuna ottimizzazione

```powershell
python -m zeroai.cli merge --mode model --base ./base --models ./e1 ./e2 `
  --method dare_ties --weights 0.2 0.8 0.8 0.2 --dare-density 0.4 --blocks 2
```

È una scorciatoia di `run` con `optimizer=none`, `write_checkpoints=True` e `merge_backend=native` (imposti forzatamente dal comando). Utile per materializzare subito una configurazione che conosci già.

### 13.4 `doctor` - diagnostica ambiente

```powershell
python -m zeroai.cli doctor
```

Stampa JSON con versione di python, torch, transformers, accelerate, safetensors, optuna, yaml; `cuda_available`, `gpu`, `vram_gb`; `mergekit_in_process`.

---

## 14. Esempi pratici

### 14.1 Verificare che l'installazione funzioni

```powershell
python -m zeroai.cli doctor
python -m zeroai.cli run --mode mock --optimizer genetic --population 4 --generations 1 --blocks 2 --run-name smoke --log-level WARNING
```

Se esce con code 0 e produce i 5 artefatti, l'intero loop (genere -> merge -> valuta -> ottimizza -> report) è funzionante.

### 14.2 Confrontare due strategie di ricerca

```powershell
python -m zeroai.cli run --mode mock --optimizer optuna --trials 40 --blocks 4 --seed 1337 --run-name optuna40
python -m zeroai.cli run --mode mock --optimizer genetic --population 20 --generations 8 --blocks 4 --seed 1337 --run-name ga20x8
```

Confronta `runs/optuna40/summary.md` e `runs/ga20x8/summary.md`: best fitness, metodo vincitore, tabella di convergenza. Con `--log-level INFO` il GA stampa anche best/mean di ogni generazione.

### 14.3 Restringere lo spazio di ricerca

```powershell
# solo metodi DARE, coefficienti fra 0.2 e 1.2, drop DARE fra 0.1 e 0.6
python -m zeroai.cli run --mode mock --methods dare_linear dare_ties `
  --weight-low 0.2 --weight-high 1.2 --dare-low 0.1 --dare-high 0.6 `
  --optimizer optuna --trials 30 --blocks 4 --run-name dare_only
```

Meno geni liberi = convergenza più rapida, ma rischio di escludere la strategia migliore.

### 14.4 Partire da un genoma già noto (refinement)

```powershell
python -m zeroai.cli run --mode mock --optimizer optuna --trials 20 `
  --genome runs\smoke\best_genome.json --run-name refine
```

Il genoma seed viene messo in coda come primo trial di Optuna (o inserito nella popolazione iniziale del GA), così la ricerca parte da una regione già misurata.

### 14.5 Merge reale su checkpoint Hugging Face

Serve: una directory base completa (con `config.json`, tokenizer) e due fine-tune **della stessa famiglia**, con gli stessi nomi di tensor.

```powershell
python -m zeroai.cli run --mode model `
  --base C:/models/Qwen2.5-0.5B `
  --models C:/models/Qwen2.5-0.5B-code C:/models/Qwen2.5-0.5B-math `
  --blocks 4 --dtype bfloat16 --device cuda `
  --optimizer genetic --population 4 --generations 1 `
  --merge-backend native --write-checkpoints `
  --run-dir runs --run-name real_model_smoke --log-level WARNING
```

Cosa succede, in ordine:

1. apertura degli store `.safetensors` e intersezione dei nomi dei tensor;
2. partizione dei layer in 4 blocchi (embedding nel blocco 0, `lm_head` nell'ultimo);
3. per ogni genoma: merge tensoriale in float32 e scrittura del candidato in bfloat16;
4. valutazione con `LLMEvaluator` (10 item) sul checkpoint del candidato;
5. report + `best_mergekit.yaml`.

Per un primo smoke test economico usa `--population 4 --generations 1` (o `--trials 4`) e un modello da 0.5B.

### 14.6 Usare mergekit come motore di fusione

```powershell
python -m zeroai.cli run --mode model --base ./base --models ./e1 ./e2 `
  --merge-backend mergekit --mergekit-python .\.venv-mergekit\Scripts\python.exe `
  --mergekit-device cuda --write-checkpoints --run-name via_mergekit
```

### 14.7 Prendere il YAML e usarlo a mano con mergekit

```powershell
python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base `
  --method dare_ties --weights 0.2 0.8 0.8 0.2 --dare-density 0.4 --blocks 2 > merge.yaml

# nel venv di mergekit:
mergekit-run merge.yaml ./merged_out
```

### 14.8 Leggere i risultati

```powershell
# top 10 candidati per fitness
python -c "import pandas as pd; df=pd.read_csv('runs/smoke/candidates.csv'); print(df.nlargest(10,'fitness')[['candidate_id','method','fitness','density','dare_density']])"

# metriche del candidato migliore
python -c "import json; d=json.load(open('runs/smoke/summary.json')); print(json.dumps(d['candidates'][0]['metrics'], indent=2))"
```

Se pandas non è installato, `summary.md` contiene già la top 5 e la tabella dei coefficienti per blocco.

---

## 15. Limiti noti e troubleshooting

| Sintomo | Causa | Soluzione |
|---|---|---|
| Tutti i candidati con `error: eval: LLMEvaluator needs a merged checkpoint on disk` | `--mode model` senza `--write-checkpoints` | aggiungi `--write-checkpoints` |
| `error: eval: Can't load config for '<path>'` | il candidato non ha `config.json` perché il base model non è una directory HF completa | usa un checkpoint HF completo come `--base`: ZeroAI copia i file non-`.safetensors` del base nel candidato |
| `no tensor name is shared between base model and input models` | base ed expert non sono della stessa famiglia o hanno naming diverso | usa expert fine-tunati dallo stesso base |
| `plan.blocks does not cover N tensor(s)` | naming esotico non riconosciuto dalla partizione | aumenta `--blocks` o allinea i nomi dei tensor |
| `mergekit_in_process: false` in `doctor` | mergekit non installato nell'interprete corrente (problema noto su Python 3.14) | venv dedicato + `--mergekit-python` |
| Inferenza lentissima in modalità `model` | `--device cpu` (default) | `--device cuda` e installa `vllm` per il backend veloce |
| In modalità mock `stability` resta vicino a 0.000000 | il mondo sintetico produce norme fuse sistematicamente fuori dal range ideale del proxy | non è un bug di pipeline; per rendere la fitness più discriminante, ricalibrare `ideal_norm` / range dei pesi in `zeroai/eval/synthetic.py` |
| Fitness mock simile fra metodi diversi | spazio dei pesi troppo stretto o budget troppo basso | aumenta `--blocks`, `--trials`/`--generations`, allarga `--weight-high` |
| NaN nel merge fp16 | somma di pesi troppo alti | usa `--dtype float32` per l'export, oppure `--normalize` (già attivo di default) |

Altri limiti da conoscere:

- `--mode mock` **non valuta un LLM**: misura il task vector contro un ground truth sintetico. Serve a confrontare strategie di merge, non a misurare qualità reale.
- I coefficienti per blocco si applicano a **tutti i tensor del blocco**, non per tipo di tensor: attention (q/k/v/o) e MLP (gate/up/down) condividono lo stesso slot.
- `dare_scale` non ha flag CLI: si configura solo tramite `SearchSpace` in libreria.
- Il comando `merge` forza `--merge-backend native` e `--write-checkpoints` (ignora `--mergekit-python`).
- `runs/` contiene artefatti di esperimento: non andrebbero versionati.
- La suite di benchmark reale è volutamente piccola e offline (10 item, checker deterministici): serve a dare un segnale confrontabile fra candidati, non a sostituire una valutazione scientifica.

---

## 16. Uso come libreria Python

```python
from zeroai import (
    EvolutionaryMergePipeline,
    PipelineConfig,
    ModelSpec,
    SearchSpace,
    MergeMethod,
    DType,
)
from zeroai.genome import Genome

config = PipelineConfig(
    mode="model",                                   # "mock" | "model"
    base_model=ModelSpec(path="C:/models/Qwen2.5-0.5B"),
    models=(
        ModelSpec(path="C:/models/Qwen2.5-0.5B-code"),
        ModelSpec(path="C:/models/Qwen2.5-0.5B-math"),
    ),
    num_blocks=4,
    dtype=DType.BFLOAT16,
    device="cuda",
    seed=1337,
    run_dir="runs",
    run_name="libreria",
    write_checkpoints=True,
    merge_backend="native",                         # "native" | "mergekit" | "auto"
    optimizer="optuna",
    n_trials=30,
    search_space=SearchSpace(
        methods=(MergeMethod.TIES, MergeMethod.DARE_TIES),
        weight_low=0.2,
        weight_high=1.2,
        dare_scale_low=1.0,                         # unico gene non esposto in CLI
        dare_scale_high=2.0,
    ),
    initial_genomes=(Genome.uniform(2, 4),),        # semi della ricerca
)

result = EvolutionaryMergePipeline(config).run()

print(result.best_fitness, result.best_genome.short_repr())
for record in result.top(5):
    print(record.candidate_id, record.fitness, record.genome.method.value)
print(result.report_paths)
```

Campi di `PipelineConfig` non esposti in CLI: `domains`, `benchmark_suite`, `max_new_tokens`, `synthetic_tensors_per_block`, `synthetic_tensor_shape`, `synthetic_noise`, `initial_genomes`.

Componenti sostituibili: implementa `Evaluator` (`zeroai/eval/base.py`) per un tuo benchmark, o `MergeBackend` (`zeroai/merge/backend.py`) per un tuo motore di fusione, e passali costruendo la pipeline a mano come fanno i test. Puoi anche usare i singoli moduli da soli:

```python
from zeroai.candidate import CandidateGenerator
from zeroai.blocks import partition_tensors
from zeroai.genome import Genome
from zeroai.schemas import ModelSpec

blocks = partition_tensors(["model.layers.0.q_proj.weight", "model.layers.1.mlp.down_proj.weight"], 2)
gen = CandidateGenerator(
    base_model=ModelSpec(path="./base"),
    models=(ModelSpec(path="./e1"), ModelSpec(path="./e2")),
    blocks=blocks,
)
print(gen.yaml(Genome.uniform(2, 2), candidate_id="manual"))
```

---

## 17. Test

```powershell
python -m pytest -o addopts=''          # output completo (summary incluso)
python -m pytest                        # usa gli addopts di pytest.ini (-q)
python -m pytest tests/test_genome.py -o addopts='' -v
```

**306 test raccolti** su 12 file, tutti verdi.

| File | Cosa copre |
|---|---|
| `tests/test_algorithms.py` | trim, elect-sign, DARE, `combine`, equivalenza con la semantica mergekit, rescale di norma |
| `tests/test_blocks.py` | partizione per profondità, pinning di embedding e head, nomi sconosciuti distribuiti |
| `tests/test_candidate_yaml.py` | `MergePlan` -> YAML mergekit, inversione `1 - dare_density`, validazioni |
| `tests/test_cli.py` | parsing e comportamento di `run`, `yaml`, `merge`, `doctor` |
| `tests/test_eval_benchmarks.py` | checker numerici, esecuzione Python, checker di istruzione |
| `tests/test_eval_synthetic.py` | mondo sintetico deterministico e metriche |
| `tests/test_genome.py` | `key`, clip, crossover, mutazione, serializzazione, normalizzazione per blocco |
| `tests/test_native_backend.py` | merge tensoriale, stats di interferenza, scrittura checkpoint |
| `tests/test_optimizers.py` | Optuna e GA: budget, dedup, seed, convergenza |
| `tests/test_pipelines.py` | loop completo mock, isolamento dei fallimenti, cache, artefatti |
| `tests/test_report.py` | `best_genome.json`, `candidates.csv`, `summary.md/json` |
| `tests/test_tensor_store.py` | store in-memory e safetensors, intersezione dei nomi |

> `pytest.ini` contiene già `addopts = -q`; aggiungere un secondo `-q` può nascondere la righe di riepilogo. Per l'output normale usa `-o addopts=''`.

---

## 18. Struttura del repository

```
ZeroAI/
  pyproject.toml            # pacchetto, extras (mergekit, serve, test), entry point "zeroai"
  pytest.ini                # testpaths=tests, pythonpath=., addopts=-q
  requirements.txt          # torch, transformers, accelerate, safetensors, PyYAML, numpy, optuna, mergekit, tqdm
  .gitignore                # ignora runs/, *.safetensors, venv, cache, *.db di Optuna
  zeroai/
    __init__.py             # API pubblica + __version__
    __main__.py             # python -m zeroai
    cli.py                  # run / yaml / merge / doctor
    schemas.py              # contratti dati condivisi
    genome.py               # Genome, SearchSpace ops, normalize_weights
    blocks.py               # partizione dei tensor in blocchi
    candidate.py            # CandidateGenerator: Genome -> MergePlan -> YAML
    pipelines.py            # EvolutionaryMergePipeline, PipelineConfig, PipelineResult
    report.py               # write_report, render_summary
    merge/
      algorithms.py         # trim, elect-sign, DARE, combine, interference_energy
      backend.py            # NativeMergeBackend
      mergekit_backend.py   # YAML emitter + driver out-of-process
      mergekit_driver.py    # script eseguito nell'interprete di mergekit
      tensor_store.py       # TensorStore, InMemoryTensorStore, SafetensorsTensorStore
    eval/
      base.py               # interfaccia Evaluator
      synthetic.py          # mondo sintetico + SyntheticEvaluator
      benchmarks.py         # BenchmarkSuite, default_suite, checker
      llm.py                # LLMEvaluator (transformers / vllm)
    optimizers/
      base.py               # interfaccia Optimizer, OptimizationResult
      optuna_opt.py         # OptunaOptimizer (TPE + pruner + storage)
      genetic.py            # GeneticOptimizer
  tests/                    # 12 file, 306 test
  runs/                     # output delle run (ignorato da git)
```

---

**Versione**: `0.1.0` (`zeroai.__version__`). **Licenza**: non dichiarata nel repository.
