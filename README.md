# ZeroAI

**Evolutionary optimization of model merging: automatically search the per-layer merge weights of an LLM model merge.**

**Language**: English (this file) · [Italiano](README.it.md)

**License**: [ISC](LICENSE) — Copyright (c) 2026 Vincenzo Tilotta

ZeroAI takes a *base model* and N *expert models* (`.safetensors` checkpoints) and searches, through Optuna (TPE) or a genetic algorithm, for the best combination of:

- **merge method** (`linear`, `taskslim`, `ties`, `dare_linear`, `dare_ties`),
- **per-model × per-block coefficients** (one weight slot for each group of layers),
- **pruning strength** (magnitude trim and DARE drop),
- **rescaling** and **normalization** of the coefficients,

measuring every candidate with a multi-domain benchmark battery. The result is the optimal **genome** (the best configuration found), exported as `best_genome.json` + `best_mergekit.yaml` ready for mergekit, plus the full report of every candidate tried.

> **Honesty note.** The `mock` mode (synthetic world) and the *tensor merge* side of `model` mode are fully working and covered by tests. Evaluation on real LLMs requires real checkpoints and VRAM: it is the only link that depends on the environment, not on the code. See [Known limitations](#15-known-limitations-and-troubleshooting).

---

## Contents

1. [What problem it solves](#1-what-problem-it-solves)
2. [Quick start (60 seconds)](#2-quick-start-60-seconds)
3. [Architecture](#3-architecture)
4. [The genome: what gets optimized](#4-the-genome-what-gets-optimized)
5. [Implemented merge methods](#5-implemented-merge-methods)
6. [Blocks: the per-layer coefficients](#6-blocks-the-per-layer-coefficients)
7. [Merge backends: native and mergekit](#7-merge-backends-native-and-mergekit)
8. [Evaluation: mock and model](#8-evaluation-mock-and-model)
9. [Optimizers: Optuna and genetic algorithm](#9-optimizers-optuna-and-genetic-algorithm)
10. [The pipeline and error handling](#10-the-pipeline-and-error-handling)
11. [Artifacts produced by each run](#11-artifacts-produced-by-each-run)
12. [Installation](#12-installation)
13. [Full CLI reference](#13-full-cli-reference)
14. [Practical examples](#14-practical-examples)
15. [Known limitations and troubleshooting](#15-known-limitations-and-troubleshooting)
16. [Using ZeroAI as a Python library](#16-using-zeroai-as-a-python-library)
17. [Tests](#17-tests)
18. [Repository layout](#18-repository-layout)
19. [License](#19-license)

---

## 1. What problem it solves

Model merging fuses several fine-tunes of the same base model without retraining anything, working in the **task vector space**:

```
delta_task    = weights(expert) - weights(base)
weights(merged) = base + sum_m ( w_m * delta_m )
```

The practical problem is that **a single global weight is not enough**: a code expert excels in the mid-low layers, a reasoning expert in the deep layers, and merging everything with weight 1.0 produces **destructive interference** (the deltas cancel each other out) and an explosion of the norm of the merged task vector, which degrades the model's general capability.

ZeroAI turns these choices into an optimization problem:

| Choice that is normally manual | In ZeroAI it is a gene of the genome |
|---|---|
| Which merge algorithm to use | `method` |
| How much each expert weighs | `weights[model][block]` |
| How much pruning to apply | `density`, `dare_density` |
| How much to rescale the surviving deltas | `dare_scale` |
| Whether to normalize the coefficients | `normalize` |

The implemented techniques (magnitude trimming, elect-sign, DARE drop+rescale) mirror the semantics of mergekit 0.1.x and the **TIES-Merging** and **DARE** papers.

---

## 2. Quick start (60 seconds)

```powershell
# 1. check the environment (python, torch, CUDA, mergekit)
python -m zeroai.cli doctor

# 2. full loop in mock mode (no model required, ~1 second)
python -m zeroai.cli run --mode mock --optimizer genetic --population 6 --generations 3 --blocks 3 --run-name smoke_ga

# 3. same loop with Optuna
python -m zeroai.cli run --mode mock --optimizer optuna --trials 12 --blocks 3 --run-name smoke_optuna

# 4. generate ONLY the mergekit YAML of a manual configuration
python -m zeroai.cli yaml --mode mock --models a b --method dare_ties --weights 0.2 0.8 --blocks 2

# 5. run ONE merge (no optimization) and write the checkpoint
python -m zeroai.cli merge --mode mock --models a b --method dare_ties --density 0.6 --blocks 2 --write-checkpoints
```

Every `run` ends with the list of artifacts written:

```
artifacts:
  best_genome:     runs\smoke_ga\best_genome.json
  mergekit_yaml:   runs\smoke_ga\best_mergekit.yaml
  candidates_csv:  runs\smoke_ga\candidates.csv
  summary:         runs\smoke_ga\summary.md
  summary_json:    runs\smoke_ga\summary.json
  run_dir:         runs\smoke_ga
```

`python -m zeroai.cli ...` and `python -m zeroai ...` are equivalent. After `pip install -e .` the `zeroai` command is available as well.

---

## 3. Architecture

The flow is a **closed loop**: the optimizer proposes a genome, the pipeline executes and evaluates it, and the score goes back to the optimizer.

```
                    +--------------------------------------------------------------+
                    |          Optimizer (Optuna TPE / genetic algorithm)         |
                    |   proposes Genome = method + per-block weights + pruning     |
                    +-------------------------------+-----------------------------+
                                                    | genome
                                                    v
    N models  -->  CandidateGenerator  -->  MergePlan + mergekit YAML
    + base            |   (per-block normalization, validation)
                      v
                MergeBackend  -->  MergedArtifact
     (native torch | mergekit subprocess)   (checkpoint + merged deltas)
                      |
                      v
                 Evaluator  -->  EvaluationReport (fitness + metrics)
        (synthetic proxy | real LLM benchmark)
                      |
                      +---- fitness ----> repeat until the budget is exhausted
                                          |
                                          v
                                    write_report()
```

| Module | File | Responsibility |
|---|---|---|
| Data contracts | `zeroai/schemas.py` | `MergeMethod`, `DType`, `ModelSpec`, `BlockSpec`, `MergePlan`, `MergedArtifact`, `EvaluationReport`, `SearchSpace` |
| Chromosome | `zeroai/genome.py` | immutable `Genome`: `sample` / `uniform` / `random`, `clipped`, `crossover`, `mutate`, `key()`, (de)serialization, `normalize_weights` |
| Layer partition | `zeroai/blocks.py` | `partition_tensors`, `layer_index`, `build_block_index` |
| Module 1 - candidate | `zeroai/candidate.py` | `CandidateGenerator`: Genome -> `MergePlan` -> mergekit YAML |
| Module 2 - merge | `zeroai/merge/` | `algorithms.py` (trim, elect-sign, DARE), `backend.py` (native engine), `mergekit_backend.py` + `mergekit_driver.py`, `tensor_store.py` |
| Module 3 - evaluation | `zeroai/eval/` | `base.py` (interface), `synthetic.py` (synthetic proxy), `benchmarks.py` (test suite), `llm.py` (real inference) |
| Module 4 - optimization | `zeroai/optimizers/` | `base.py`, `optuna_opt.py`, `genetic.py` |
| Module 5 - orchestration | `zeroai/pipelines.py` | `PipelineConfig`, `EvolutionaryMergePipeline`, `PipelineResult` |
| Module 6 - report | `zeroai/report.py` | `write_report`, `render_summary` |
| Interface | `zeroai/cli.py`, `zeroai/__main__.py` | `run`, `yaml`, `merge`, `doctor` |

The modules communicate **only** through the contracts in `schemas.py`: backend, evaluator and optimizer are interchangeable and testable in isolation.

---

## 4. The genome: what gets optimized

`zeroai/genome.py` defines `Genome`, a **frozen** (immutable) dataclass: it can be used as a dictionary key, put in a set for dedup, passed to Optuna or placed in the GA population without defensive copies.

| Gene | Type | Meaning | Default range in `SearchSpace` |
|---|---|---|---|
| `method` | `MergeMethod` | fusion strategy | all 5 methods |
| `weights` | tuple `[model][block]` | coefficient of each expert for each block of layers | `0.0 .. 1.5` |
| `density` | float | **fraction of coefficients kept** by the magnitude trim (used by `taskslim`, `ties`, `dare_ties`) | `0.05 .. 1.0` |
| `dare_density` | float | **fraction of coefficients discarded** with a DARE bernoulli drop | `0.0 .. 0.9` |
| `dare_scale` | float | additional rescaling factor applied to the surviving deltas | `1.0 .. 3.0` |
| `normalize` | bool | whether to bring the coefficients of each block back to a sum of 1 | `True` |
| `dtype` | `DType` | write dtype of the checkpoint | `float16` |

Main methods of `Genome`:

- `key()` - stable identity string (method + scalar parameters + weights rounded to 6 decimals) used to **deduplicate** evaluations;
- `sample()` / `Genome.random()` - uniform draw inside the `SearchSpace`;
- `Genome.uniform(n_models, n_blocks)` - every expert with weight `1/n_models` on every block (the "naive" merge);
- `clipped(space)` - brings every gene back inside the limits (used after crossover and mutation);
- `crossover(other, rng, alpha=0.5)` - **BLX-alpha blend crossover** on the continuous genes + uniform crossover (50/50) on the discrete genes (`method`, `normalize`);
- `mutate(rng, rate, sigma)` - gaussian mutation (sigma) on the continuous genes and re-draw of the `method` gene;
- `to_dict()` / `from_dict()` - JSON serialization: this is the format of `best_genome.json` and of `--genome`;
- `short_repr()` - readable line used in logs.

> **Careful with `density` vs `dare_density`.** `density` is the fraction **kept**, `dare_density` is the fraction **discarded**. In the mergekit YAML both end up in mergekit's `density` key (which is always the kept fraction), so for the DARE methods ZeroAI writes `1 - dare_density`.

### Weight normalization

`normalize_weights()` in `zeroai/genome.py` normalizes **per column, that is per block** (not per model as mergekit does): the sum of the absolute values of the model coefficients inside each block is brought to 1. This keeps the norm of the merged task vector under control block by block, which is the main cause of the collapse of general capability. If you want "pass-through" weights with no rescaling, use `--no-normalize`.

---

## 5. Implemented merge methods

All of them in `zeroai/merge/algorithms.py`. The arithmetic accumulation always happens in **float32** (`WORK_DTYPE`) regardless of the storage dtype, as mergekit does internally: this is what makes fp16 merges stable.

| Method | Per-tensor pipeline | Genes used |
|---|---|---|
| `linear` | weighted sum of the deltas | `weights` |
| `taskslim` | magnitude trim (keeps `density`) -> weighted sum | `weights`, `density` |
| `ties` | trim -> **elect-sign** (weighted majority sign) -> discard the coefficients disagreeing with the sign -> weighted sum | `weights`, `density` |
| `dare_linear` | bernoulli **DARE drop** (keeps `1 - dare_density`, L2 rescale) x `dare_scale` -> weighted sum | `weights`, `dare_density`, `dare_scale` |
| `dare_ties` | DARE drop+rescale -> trim -> elect-sign -> discard the sign outliers -> weighted sum | all of them |

Implementation details:

- `magnitude_trim(tensor, density)` - keeps the `density * numel` coefficients with the largest magnitude (`topk` over `abs`).
- `bernoulli_drop(tensor, density)` - bernoulli mask; on CPU with fp16 it upcasts to float32 because `torch.bernoulli` does not support fp16 on CPU.
- `_rescaled_masked_tensor` - after masking it restores the original norm (`l1`, `l2`, `linf`) exactly like mergekit's `rescale_norm` (the "REscale" step of the DARE paper). For DARE, ZeroAI uses `l2`; for the trim it uses `None` (no rescale).
- `elect_sign(tensors, weights)` - accumulates `weight * tensor` and takes its sign; zero-sum cases are resolved towards `+1` so that a coefficient is not silently discarded.
- `interference_energy(contributions)` - returns `(cross_energy, self_energy)`: `cross_energy` is the sum of the cross dot products between contributions from different models. Negative values = destructive interference. It ends up in the `stats` of every candidate and therefore in `candidates.csv` / `summary.json`.

Every method has an equivalent name in the mergekit YAML (`MergeMethod.mergekit_method`): the names are identical (`linear`, `taskslim`, `ties`, `dare_linear`, `dare_ties`).

---

## 6. Blocks: the per-layer coefficients

A **block** is a contiguous group of layers that receives its own fusion coefficient: it is the structure that turns "one global weight per model" into "one weight per model per depth zone".

`zeroai/blocks.py::partition_tensors(names, num_blocks)`:

1. deduplicates and sorts the tensor names;
2. extracts the layer index with three regexes: `...layers.<n>.`, `...h.<n>.` / `block.<n>.` / `decoder_layer.<n>.`, and the internal pattern `block_<n>.` used by the synthetic world;
3. distributes the layers over the blocks proportionally to depth: `block = floor((idx + 1) / (max_idx + 1) * num_blocks)`;
4. **pins the embeddings into the first block** (`embed_tokens`, `word_embeddings`, `wte`, `embeddings.`, `tok_embeddings`) and **the output heads into the last one** (`lm_head`, `output.`, `final_dense`, `classifier`);
5. names whose depth cannot be inferred (exotic checkpoints) are distributed evenly across the blocks instead of piling up in block 0.

`--blocks N` controls how many coefficient slots each model gets: with 2 models and `--blocks 4` the genome has 8 free weights. With `--blocks 1` the problem degenerates into classic single-weight linear merging.

`MergePlan.validate()` checks that the weight matrix has exactly `n_models` rows and `n_blocks` columns, that `0 < density <= 1`, `0 <= dare_density < 1` and `dare_scale > 0`. The native backend additionally checks that `plan.blocks` covers **all** the tensors shared between base and experts.

---

## 7. Merge backends: native and mergekit

### `NativeMergeBackend` (default, always available)

`zeroai/merge/backend.py`. In-process torch engine:

- reads the tensors from the `TensorStore`s (in-memory for mock mode, `SafetensorsTensorStore` for real checkpoints);
- considers only the **shared tensor names** between base and experts (`common_tensor_names`);
- computes the `expert - base` deltas and applies `combine(...)` to each tensor with the coefficient of the corresponding block;
- produces `base + merged_delta` written in the requested `dtype`;
- accumulates the interference `stats` (`interference_cross_energy`, `interference_self_energy`, `interference_ratio`, `merged_delta_norm`, `base_norm`);
- with `save_output=True` it writes `<out>/model.safetensors` (metadata `format: pt`) and **copies every non-`.safetensors` file from the base model directory** (`config.json`, tokenizer, chat template, etc.) into the candidate directory, so the candidate can be loaded by `transformers`.

`SafetensorsTensorStore` loads **lazily and with caching**: only the tensors actually touched get materialized, and they stay in cache for the whole evolutionary loop.

### `MergekitBackend` (out-of-process)

`zeroai/merge/mergekit_backend.py` + `zeroai/merge/mergekit_driver.py`.

mergekit 0.1.x requires `huggingface_hub>=1.31`, while transformers 4.x requires `huggingface_hub<1.0`: installing them in the same environment breaks transformers. That is why ZeroAI **does not import mergekit into its own interpreter**:

- the **YAML is always generated** by the `CandidateGenerator` (and is testable without mergekit installed);
- execution happens by launching `python -m zeroai.merge.mergekit_driver --config <yaml> --out <dir> --device <dev>` with the interpreter given by `--mergekit-python`, that is a **venv dedicated to mergekit**;
- the driver imports `mergekit.merge.run_merge` with `copy_tokenizer=True` and returns a JSON with the timings; exit codes: `3` = mergekit not available, `4` = invalid configuration.

`--merge-backend`:

| Value | Behaviour |
|---|---|
| `native` | internal torch engine (**CLI default**) |
| `mergekit` | always the external driver; fails if mergekit cannot be reached |
| `auto` | tries mergekit in-process, otherwise falls back to native (the `PipelineConfig` default when ZeroAI is used as a library; in `mock` mode it uses native anyway) |

`MergekitBackend.is_available()` (used by `doctor`) runs `python -c "import mergekit.merge"` in the given interpreter.

### Generated YAML

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

This is the real output of `python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base --method dare_ties --weights 0.2 0.8 0.8 0.2 --density 0.6 --dare-density 0.4 --dare-scale 1.2 --blocks 2`.

How to read it:

- `weight` is a **list of interpolated weights** in mergekit's format: one value per block, linearly interpolated by mergekit over the normalized layer position. This is mergekit's native mechanism for layer-wise weights.
- In `--mode mock` the emitted paths are `synthetic://<label>`; in `--mode model` they are exactly the paths you pass to `--models` / `--base`.
- `density` inside each model's `parameters` is always the **kept** fraction: for the DARE methods ZeroAI writes `1 - dare_density` (here 0.4 discarded -> 0.6 kept). `rescale_norm: l2` appears only in the DARE methods, `normalize: true` only when the `normalize` gene is active.
- The global `parameters` block is mergekit's default for models that do not declare those keys.

> Real normalization example: with `--weights 0.2 0.8` and normalization active (the default) the block column is rescaled to `[0.5, 0.5]`. If you want 0.2/0.8 to stay as they are, pass `--no-normalize`.

---

## 8. Evaluation: mock and model

The mode chosen with `--mode` determines **how a candidate is measured**.

### 8.1 `--mode mock` (default): deterministic synthetic proxy

`zeroai/eval/synthetic.py::build_synthetic_world()` builds a reproducible "merge world" (same `--seed` -> same matrices):

- a base model of small tensors (`synthetic_tensor_shape` defaults to `32x32`, `synthetic_tensors_per_block=2`);
- one "truth" vector per domain (`reasoning` weight 0.30, `code` 0.25, `math` 0.20, `general` 0.25), built as a **depth profile** (a gaussian centred on `t=0.65` for reasoning, `0.35` for code, `0.5` for math/general) plus a **shared core** across domains, to model capabilities that are not orthogonal in weight space;
- N experts whose domain coverage rotates (each expert is strong on a subset of domains);
- **deliberate sign conflicts** on a fraction of (expert, block) pairs: exactly the case elect-sign has to resolve;
- independent gaussian noise per expert (`synthetic_noise=0.35`).

Scoring requires no inference: it evaluates the **merged task vector** (`artifact.deltas`) produced by the native backend.

| Metric | How it is computed |
|---|---|
| `capability` | cosine similarity between the merged vector of each block and the domain's truth vector, **weighted by the domain's depth profile**, averaged over domains |
| `purity` | fraction of the merged delta's energy that stays **inside the capability subspace**: least-squares projection of the merged vector onto the truth vectors of all domains; `1 - residual_energy / total_energy` |
| `stability` | `1 - abs(norm(merged_delta) - ideal_norm) / ideal_norm`, where `ideal_norm` is the norm of the domain-weighted "truth" delta: it penalizes the over-amplification that destroys general capability |
| `cooperation` | `1 - destructive interference` between the weighted expert contributions (cross energy over the experts' original deltas) |
| `domain_<name>` | score of the single domain (`reasoning`, `code`, `math`, `general`) |

**Fitness** = `0.55 * capability + 0.15 * purity + 0.15 * stability + 0.15 * cooperation` (weights in `SyntheticBenchmark.fitness_weights`).

What it is good for: validating the architecture, testing the optimizers, checking that the per-block coefficients really matter, and developing without a GPU or downloads.

### 8.2 `--mode model`: real benchmark on merged checkpoints

`zeroai/eval/llm.py::LLMEvaluator` loads the candidate checkpoint and answers a battery of offline, deterministic items (`zeroai/eval/benchmarks.py::default_suite`, 10 items):

| Domain | Weight | Items | Checker |
|---|---|---|---|
| `reasoning` | 0.35 | 4 items: average speed, proportions, sequences, percentages | `NumericAnswerChecker` (parses the last number, tolerance 1e-4) |
| `code` | 0.35 | 3 items: `sum_squares`, `fizzbuzz`, `reverse_words` | `PythonExecChecker`: extracts the code fence, appends the `assert` statements and runs it in a subprocess with a 10 s timeout |
| `general` | 0.30 | explanation, tone rewriting, list formatting | `InstructionChecker`: length band, required markers, absence of refusal phrases |

**Fitness** = weighted average of the domain scores (1 = every item of the domain passed). The additional metric `benchmark_pass_rate` (plain average over all items, weight 0) is recorded as well.

Inference backend (`_resolve_backend`):

- `vllm` if installed (much faster over the many-candidate loop), otherwise
- `transformers` (`AutoModelForCausalLM` + `AutoTokenizer`, `device_map="auto"`, greedy generation because `temperature=0`).

The model stays loaded until the next candidate changes path (`_loaded_from`); `close()` releases the VRAM at the end of the run.

> **Important constraint.** `LLMEvaluator` requires `artifact.output_path`, so in `--mode model` real evaluation works **only with `--write-checkpoints`**. Without it, every candidate is marked `eval: LLMEvaluator needs a merged checkpoint on disk` (fitness 0) and the search cannot progress.

---

## 9. Optimizers: Optuna and genetic algorithm

### Optuna (`--optimizer optuna`, default)

`zeroai/optimizers/optuna_opt.py`. It explores the mixed categorical/continuous space:

- `suggest_categorical("method", ...)`
- `suggest_float("weight_m{model}_b{block}", weight_low, weight_high)` for every model/block pair
- `suggest_float("density")`, `suggest_float("dare_density")`, `suggest_float("dare_scale")`
- `suggest_categorical("normalize", [True, False])`

Configuration: `TPESampler(seed=seed, n_startup_trials=min(10, n_trials))`, `MedianPruner(n_startup_trials=5)` (can be disabled with `prune=False`), direction `maximize`, `gc_after_trial=True`, an in-memory study or one on `storage` (e.g. `sqlite:///zeroai.db` with `load_if_exists`) to resume an interrupted search. Genomes passed in `initial_genomes` are queued with `enqueue_trial` (`_flatten_genome`), so the supplied seed is always evaluated.

### Genetic algorithm (`--optimizer genetic`)

`zeroai/optimizers/genetic.py`.

| Parameter | Class default | CLI default | Notes |
|---|---|---|---|
| `population_size` | 24 | `--population` 20 | minimum 4 |
| `generations` | 12 | `--generations` 8 | minimum 1 |
| `mutation_rate` | 0.25 | - | probability per gene |
| `mutation_sigma` | 0.15 | - | gaussian deviation on the weights |
| `crossover_rate` | 0.7 | - | blend crossover |
| `method_mutation_rate` | 0.15 | - | independent re-draw of the `method` gene |
| `tournament_size` | 3 | - | tournament selection |
| `elite_keep` | 2 | - | elitism |

Mechanism: initial population (seed + random, deduplicated by `key()`), then for each generation elitism -> tournament -> crossover -> mutation -> method re-draw -> `clipped()` -> evaluation with an internal cache.

Key difference from Optuna: the GA **also mutates the merge-method gene**, so it can escape a basin where a whole family of strategies (for example all the TIES variants) is stuck. It is also the fallback when optuna is not installed.

### `--optimizer none`

Evaluates a single genome (the one supplied with `--genome`, or built from `--method`/`--weights`, otherwise a random one) and still produces all the reports. This is the mode used internally by the `yaml` and `merge` commands.

---

## 10. The pipeline and error handling

`zeroai/pipelines.py::EvolutionaryMergePipeline` contains no optimization logic: it wires the four modules together and handles caching, persistence and resilience.

### `PipelineConfig` (all fields)

| Field | Default | Notes |
|---|---|---|
| `mode` | `"mock"` | `"mock"` or `"model"` |
| `base_model` / `models` | `None` / `()` | required in `model` mode (validated in `__post_init__`) |
| `num_blocks` | 4 | >= 1 |
| `dtype` | `DType.FLOAT16` | `float16` / `bfloat16` / `float32` |
| `seed` | 1337 | determinism of the synthetic world, DARE and the sampler |
| `run_dir` / `run_name` | `"runs"` / `"run"` | the actual directory is `run_dir/run_name`, made absolute and created automatically |
| `search_space` | `SearchSpace()` | gene limits |
| `write_checkpoints` | `False` | writes the candidate checkpoints |
| `merge_backend` | `"auto"` | `"native"`, `"mergekit"`, `"auto"` |
| `mergekit_python` / `mergekit_device` | `None` / `"cpu"` | interpreter and device of the mergekit driver |
| `device` | `"cpu"` | device of the merge and of the evaluator |
| `optimizer` | `"optuna"` | `"optuna"`, `"genetic"`, `"none"` |
| `n_trials` / `population_size` / `generations` | 40 / 20 / 8 | search budget |
| `initial_genomes` | `()` | search seeds |
| `domains` | `DEFAULT_DOMAINS` | domains of the synthetic world |
| `benchmark_suite` | `None` | custom suite for `LLMEvaluator` |
| `max_new_tokens` | 256 | generation of the real evaluator |
| `synthetic_tensors_per_block` / `synthetic_tensor_shape` / `synthetic_noise` | 2 / `(32, 32)` / 0.35 | parameters of the mock world |

### `prepare()`

- **mock**: builds the synthetic world and creates the fictitious `ModelSpec`s `synthetic://<label>`. If `--models` is empty, 3 experts are generated (`expert_0..2`); if `--models` is given, the **names** of the paths become the expert labels (the paths are not read).
- **model**: opens a `SafetensorsTensorStore` for the base and for each expert, computes the **shared tensor names** and partitions the blocks over that set. If no tensor is shared, the run fails immediately with a clear error.

### Per-candidate resilience

`_score()` wraps every candidate:

| Failure | Outcome |
|---|---|
| exception during the merge | `CandidateRecord(error="merge: ...")`, fitness 0, search continues |
| exception during evaluation | `error="eval: ..."`, fitness 0, search continues |
| NaN / inf metrics | `error="non-finite fitness"`, fitness 0 |

Failed candidates stay in the report but are **excluded** from `PipelineResult.top()` and from the "Top candidates" block of `summary.md`: a broken candidate cannot make an invalid configuration "win".

### Deduplication

`_score_cache` is indexed by `Genome.key()`: if the optimizer re-proposes an already-seen configuration (very frequent with elitism and the pruner), neither the merge nor the evaluation is repeated. Optuna and the GA each have their own analogous internal cache.

---

## 11. Artifacts produced by each run

Every run writes into `runs/<run-name>/`:

| File | Content |
|---|---|
| `best_genome.json` | serialized optimal genome (method, weights, density, dare_density, dare_scale, normalize, dtype), best fitness, duration, number of candidates and failures, domain weights, block sizes, and the `optimization` section with the **convergence history** |
| `best_mergekit.yaml` | mergekit configuration of the optimal genome, ready to run |
| `candidates.csv` | one row per candidate: `candidate_id, method, fitness, density, dare_density, dare_scale, normalize, weight_m{m}_b{b}..., <all metrics>, merge_seconds, eval_seconds, error` |
| `summary.md` | human-readable report: run header, best genome, per-block coefficient table, top 5 candidates, metrics of the best candidate, convergence table |
| `summary.json` | complete machine-readable dump (`PipelineResult.to_dict()`): all candidates with genome, metrics, interference `stats`, failures, blocks, domains, optimization |
| `candidates/cand_0001/`, `cand_0002/`, ... | only with `--write-checkpoints`: the candidate's `model.safetensors` + the auxiliary files copied from the base model |

The interference `stats` available in `summary.json` for each candidate are `tensors`, `blocks_used`, `interference_cross_energy`, `interference_self_energy`, `interference_ratio`, `merged_delta_norm`, `base_norm`: they explain **why** a genome won (for example a lower interference ratio at equal capability).

---

## 12. Installation

Requirement: **Python >= 3.10** (declared in `pyproject.toml`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # torch, transformers, accelerate, safetensors, PyYAML, numpy, optuna, tqdm
pip install -e .                     # makes the "zeroai" command available
```

`pyproject.toml` also declares the extras:

```powershell
pip install -e ".[test]"      # pytest
pip install -e ".[serve]"     # vllm: fast inference backend for model mode
pip install -e ".[mergekit]"  # mergekit, only if you know you have no conflicts with transformers
```

### mergekit in a separate venv (recommended)

mergekit 0.1.x and transformers 4.x have incompatible constraints on `huggingface_hub`, and mergekit does not install cleanly on Python 3.14. ZeroAI still works: it generates the YAML itself and mergekit execution goes through an external interpreter.

```powershell
python -m venv .venv-mergekit
.\.venv-mergekit\Scripts\Activate.ps1
pip install mergekit
deactivate

# then, from the main environment:
python -m zeroai.cli run --mode model --base ./base --models ./e1 ./e2 `
  --merge-backend mergekit `
  --mergekit-python .\.venv-mergekit\Scripts\python.exe `
  --mergekit-device cuda --write-checkpoints --run-name via_mergekit
```

### Environment check

```powershell
python -m zeroai.cli doctor
```

Example output (machine with a CUDA GPU):

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

`mergekit_in_process: false` **is not an error**: it only means the in-process mergekit backend is not available and that you should use `native` or a dedicated venv.

---

## 13. Full CLI reference

```
usage: zeroai [-h] {run,yaml,merge,doctor} ...

  run     run the closed evolutionary merge loop
  yaml    emit the mergekit YAML for a genome
  merge   execute one merge, no optimisation
  doctor  report environment / GPU / mergekit readiness
```

### 13.1 `run` - the full evolutionary loop

```powershell
python -m zeroai.cli run [flag]
```

**General flags**

| Flag | Default | Description |
|---|---|---|
| `--mode {mock,model}` | `mock` | `mock` = synthetic world, no files required; `model` = real `.safetensors` checkpoints |
| `--models M1 M2 ...` | empty | expert paths. In `mock` they only provide the labels (if empty, 3 synthetic experts); in `model` they are the checkpoints to fuse |
| `--base PATH` | `None` | base model (required in `model`) |
| `--blocks N` | 4 | number of coefficient slots per model |
| `--dtype {float16,bfloat16,float32}` | `float16` | dtype of the written checkpoint and of the inference |
| `--device DEVICE` | `cpu` | device of the merge and of the evaluator (`cuda` for real models) |
| `--seed INT` | 1337 | determinism |
| `--run-dir DIR` | `runs` | root of the runs |
| `--run-name NAME` | `run` | `runs/NAME` subdirectory |
| `--log-level LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, ... |

**Search flags**

| Flag | Default | Description |
|---|---|---|
| `--optimizer {optuna,genetic,none}` | `optuna` | search engine |
| `--trials N` | 40 | Optuna budget |
| `--population N` | 20 | GA population size |
| `--generations N` | 8 | GA generations |
| `--methods linear ties ...` | all of them | restricts the `method` gene (accepts the 5 method names) |
| `--weight-low` / `--weight-high` | 0.0 / 1.5 | range of the per-block coefficients |
| `--density-low` / `--density-high` | 0.05 / 1.0 | range of the fraction kept by the trim |
| `--dare-low` / `--dare-high` | 0.0 / 0.9 | range of the fraction discarded by DARE |
| `--genome FILE.json` | `None` | seed genome read from a file (for example a `best_genome.json` from a previous run) |

> `dare_scale` (range 1.0-3.0) **has no CLI flag**: it is configured only through `SearchSpace` when ZeroAI is used as a library.

**Explicit genome flags** (useless with an active optimizer, essential for `yaml` and `merge`)

| Flag | Default | Description |
|---|---|---|
| `--method NAME` | `None` | fixed merge method |
| `--weights W...` | `None` | coefficients: `len(blocks)` values shared by all models, or `blocks*models` values in model-major order |
| `--density X` | 1.0 | fraction kept by the trim |
| `--dare-density X` | 0.0 | fraction discarded by DARE |
| `--dare-scale X` | 1.0 | rescaling of the surviving deltas |
| `--normalize` / `--no-normalize` | normalization on | bring the sum of each block's coefficients back to 1 |

**Merge backend flags**

| Flag | Default | Description |
|---|---|---|
| `--merge-backend {auto,native,mergekit}` | `native` | fusion engine |
| `--mergekit-python PATH` | current interpreter | python of the venv that contains mergekit |
| `--mergekit-device DEVICE` | `cpu` | device used by the mergekit driver |
| `--write-checkpoints` | off | writes `runs/NAME/candidates/cand_XXXX/model.safetensors` for every candidate (**required for real evaluation**) |

### 13.2 `yaml` - print the mergekit configuration

```powershell
python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base `
  --method ties --weights 0.3 0.7 --blocks 2
```

It accepts the same flags as `run` (including the search options, used only to build the `SearchSpace`), forces `optimizer=none` and prints the YAML to stdout without executing any merge. If you provide neither `--genome` nor `--method`/`--weights`, it uses `Genome.uniform` (weights `1/n_models`).

### 13.3 `merge` - a single merge, no optimization

```powershell
python -m zeroai.cli merge --mode model --base ./base --models ./e1 ./e2 `
  --method dare_ties --weights 0.2 0.8 0.8 0.2 --dare-density 0.4 --blocks 2
```

It is a shortcut for `run` with `optimizer=none`, `write_checkpoints=True` and `merge_backend=native` (forced by the command). Useful to materialize immediately a configuration you already know.

### 13.4 `doctor` - environment diagnostics

```powershell
python -m zeroai.cli doctor
```

Prints JSON with the versions of python, torch, transformers, accelerate, safetensors, optuna, yaml; `cuda_available`, `gpu`, `vram_gb`; `mergekit_in_process`.

---

## 14. Practical examples

### 14.1 Check that the installation works

```powershell
python -m zeroai.cli doctor
python -m zeroai.cli run --mode mock --optimizer genetic --population 4 --generations 1 --blocks 2 --run-name smoke --log-level WARNING
```

If it exits with code 0 and produces the 5 artifacts, the whole loop (generate -> merge -> evaluate -> optimize -> report) is working.

### 14.2 Compare two search strategies

```powershell
python -m zeroai.cli run --mode mock --optimizer optuna --trials 40 --blocks 4 --seed 1337 --run-name optuna40
python -m zeroai.cli run --mode mock --optimizer genetic --population 20 --generations 8 --blocks 4 --seed 1337 --run-name ga20x8
```

Compare `runs/optuna40/summary.md` and `runs/ga20x8/summary.md`: best fitness, winning method, convergence table. With `--log-level INFO` the GA also prints best/mean for every generation.

### 14.3 Narrow the search space

```powershell
# DARE methods only, coefficients between 0.2 and 1.2, DARE drop between 0.1 and 0.6
python -m zeroai.cli run --mode mock --methods dare_linear dare_ties `
  --weight-low 0.2 --weight-high 1.2 --dare-low 0.1 --dare-high 0.6 `
  --optimizer optuna --trials 30 --blocks 4 --run-name dare_only
```

Fewer free genes = faster convergence, but the risk of excluding the best strategy.

### 14.4 Start from a known genome (refinement)

```powershell
python -m zeroai.cli run --mode mock --optimizer optuna --trials 20 `
  --genome runs\smoke\best_genome.json --run-name refine
```

The seed genome is queued as Optuna's first trial (or inserted into the GA's initial population), so the search starts from a region that has already been measured.

### 14.5 Real merge on Hugging Face checkpoints

You need: a complete base directory (with `config.json`, tokenizer) and two fine-tunes **of the same family**, with the same tensor names.

```powershell
python -m zeroai.cli run --mode model `
  --base C:/models/Qwen2.5-0.5B `
  --models C:/models/Qwen2.5-0.5B-code C:/models/Qwen2.5-0.5B-math `
  --blocks 4 --dtype bfloat16 --device cuda `
  --optimizer genetic --population 4 --generations 1 `
  --merge-backend native --write-checkpoints `
  --run-dir runs --run-name real_model_smoke --log-level WARNING
```

What happens, in order:

1. the `.safetensors` stores are opened and the tensor names are intersected;
2. the layers are partitioned into 4 blocks (embeddings in block 0, `lm_head` in the last one);
3. for each genome: tensor merge in float32 and candidate written in bfloat16;
4. evaluation with `LLMEvaluator` (10 items) on the candidate checkpoint;
5. report + `best_mergekit.yaml`.

For a cheap first smoke test use `--population 4 --generations 1` (or `--trials 4`) and a 0.5B model.

### 14.6 Use mergekit as the fusion engine

```powershell
python -m zeroai.cli run --mode model --base ./base --models ./e1 ./e2 `
  --merge-backend mergekit --mergekit-python .\.venv-mergekit\Scripts\python.exe `
  --mergekit-device cuda --write-checkpoints --run-name via_mergekit
```

### 14.7 Take the YAML and run it manually with mergekit

```powershell
python -m zeroai.cli yaml --mode mock --models ./e1 ./e2 --base ./base `
  --method dare_ties --weights 0.2 0.8 0.8 0.2 --dare-density 0.4 --blocks 2 > merge.yaml

# in the mergekit venv:
mergekit-run merge.yaml ./merged_out
```

### 14.8 Read the results

```powershell
# top 10 candidates by fitness
python -c "import pandas as pd; df=pd.read_csv('runs/smoke/candidates.csv'); print(df.nlargest(10,'fitness')[['candidate_id','method','fitness','density','dare_density']])"

# metrics of the best candidate
python -c "import json; d=json.load(open('runs/smoke/summary.json')); print(json.dumps(d['candidates'][0]['metrics'], indent=2))"
```

If pandas is not installed, `summary.md` already contains the top 5 and the per-block coefficient table.

---

## 15. Known limitations and troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| All candidates with `error: eval: LLMEvaluator needs a merged checkpoint on disk` | `--mode model` without `--write-checkpoints` | add `--write-checkpoints` |
| `error: eval: Can't load config for '<path>'` | the candidate has no `config.json` because the base model is not a complete HF directory | use a complete HF checkpoint as `--base`: ZeroAI copies the base's non-`.safetensors` files into the candidate |
| `no tensor name is shared between base model and input models` | base and experts are not of the same family or use different naming | use experts fine-tuned from the same base |
| `plan.blocks does not cover N tensor(s)` | exotic naming not recognized by the partition | increase `--blocks` or align the tensor names |
| `mergekit_in_process: false` in `doctor` | mergekit is not installed in the current interpreter (known problem on Python 3.14) | dedicated venv + `--mergekit-python` |
| Very slow inference in `model` mode | `--device cpu` (the default) | `--device cuda` and install `vllm` for the fast backend |
| In mock mode `stability` stays close to 0.000000 | the synthetic world produces merged norms systematically outside the proxy's ideal range | this is not a pipeline bug; to make fitness more discriminative, recalibrate `ideal_norm` / the weight ranges in `zeroai/eval/synthetic.py` |
| Mock fitness similar across different methods | weight space too narrow or budget too low | increase `--blocks`, `--trials`/`--generations`, widen `--weight-high` |
| NaN in an fp16 merge | sum of weights that are too high | use `--dtype float32` for the export, or `--normalize` (already on by default) |

Other limitations to know:

- `--mode mock` **does not evaluate an LLM**: it measures the task vector against a synthetic ground truth. It is meant for comparing merge strategies, not for measuring real quality.
- The per-block coefficients apply to **every tensor in the block**, not per tensor type: attention (q/k/v/o) and MLP (gate/up/down) share the same slot.
- `dare_scale` has no CLI flag: it is configured only through `SearchSpace` in library use.
- The `merge` command forces `--merge-backend native` and `--write-checkpoints` (it ignores `--mergekit-python`).
- `runs/` contains experiment artifacts: they should not be versioned.
- The real benchmark suite is deliberately small and offline (10 items, deterministic checkers): it provides a comparable signal between candidates, it does not replace a scientific evaluation.

---

## 16. Using ZeroAI as a Python library

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
    run_name="library",
    write_checkpoints=True,
    merge_backend="native",                         # "native" | "mergekit" | "auto"
    optimizer="optuna",
    n_trials=30,
    search_space=SearchSpace(
        methods=(MergeMethod.TIES, MergeMethod.DARE_TIES),
        weight_low=0.2,
        weight_high=1.2,
        dare_scale_low=1.0,                         # the only gene not exposed in the CLI
        dare_scale_high=2.0,
    ),
    initial_genomes=(Genome.uniform(2, 4),),        # search seeds
)

result = EvolutionaryMergePipeline(config).run()

print(result.best_fitness, result.best_genome.short_repr())
for record in result.top(5):
    print(record.candidate_id, record.fitness, record.genome.method.value)
print(result.report_paths)
```

`PipelineConfig` fields not exposed in the CLI: `domains`, `benchmark_suite`, `max_new_tokens`, `synthetic_tensors_per_block`, `synthetic_tensor_shape`, `synthetic_noise`, `initial_genomes`.

Replaceable components: implement `Evaluator` (`zeroai/eval/base.py`) for your own benchmark, or `MergeBackend` (`zeroai/merge/backend.py`) for your own fusion engine, and pass them by building the pipeline by hand as the tests do. You can also use the individual modules on their own:

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

## 17. Tests

```powershell
python -m pytest -o addopts=''          # full output (summary included)
python -m pytest                        # uses the addopts from pytest.ini (-q)
python -m pytest tests/test_genome.py -o addopts='' -v
```

**306 tests collected** across 12 files, all green.

| File | What it covers |
|---|---|
| `tests/test_algorithms.py` | trim, elect-sign, DARE, `combine`, equivalence with mergekit semantics, norm rescaling |
| `tests/test_blocks.py` | depth partition, embedding/head pinning, distribution of unknown names |
| `tests/test_candidate_yaml.py` | `MergePlan` -> mergekit YAML, the `1 - dare_density` inversion, validations |
| `tests/test_cli.py` | parsing and behaviour of `run`, `yaml`, `merge`, `doctor` |
| `tests/test_eval_benchmarks.py` | numeric checkers, Python execution, instruction checker |
| `tests/test_eval_synthetic.py` | deterministic synthetic world and metrics |
| `tests/test_genome.py` | `key`, clipping, crossover, mutation, serialization, per-block normalization |
| `tests/test_native_backend.py` | tensor merge, interference stats, checkpoint writing |
| `tests/test_optimizers.py` | Optuna and GA: budget, dedup, seeding, convergence |
| `tests/test_pipelines.py` | full mock loop, failure isolation, cache, artifacts |
| `tests/test_report.py` | `best_genome.json`, `candidates.csv`, `summary.md/json` |
| `tests/test_tensor_store.py` | in-memory and safetensors stores, name intersection |

> `pytest.ini` already contains `addopts = -q`; adding a second `-q` can hide the summary lines. For normal output use `-o addopts=''`.

---

## 18. Repository layout

```
ZeroAI/
  pyproject.toml            # package, extras (mergekit, serve, test), "zeroai" entry point
  pytest.ini                # testpaths=tests, pythonpath=., addopts=-q
  requirements.txt          # torch, transformers, accelerate, safetensors, PyYAML, numpy, optuna, mergekit, tqdm
  .gitignore                # ignores runs/, *.safetensors, venv, cache, Optuna *.db
  LICENSE                   # ISC license, Copyright (c) 2026 Vincenzo Tilotta
  zeroai/
    __init__.py             # public API + __version__
    __main__.py             # python -m zeroai
    cli.py                  # run / yaml / merge / doctor
    schemas.py              # shared data contracts
    genome.py               # Genome, SearchSpace ops, normalize_weights
    blocks.py               # partition of tensors into blocks
    candidate.py            # CandidateGenerator: Genome -> MergePlan -> YAML
    pipelines.py            # EvolutionaryMergePipeline, PipelineConfig, PipelineResult
    report.py               # write_report, render_summary
    merge/
      algorithms.py         # trim, elect-sign, DARE, combine, interference_energy
      backend.py            # NativeMergeBackend
      mergekit_backend.py   # YAML emitter + out-of-process driver
      mergekit_driver.py    # script executed inside the mergekit interpreter
      tensor_store.py       # TensorStore, InMemoryTensorStore, SafetensorsTensorStore
    eval/
      base.py               # Evaluator interface
      synthetic.py          # synthetic world + SyntheticEvaluator
      benchmarks.py         # BenchmarkSuite, default_suite, checkers
      llm.py                # LLMEvaluator (transformers / vllm)
    optimizers/
      base.py               # Optimizer interface, OptimizationResult
      optuna_opt.py         # OptunaOptimizer (TPE + pruner + storage)
      genetic.py            # GeneticOptimizer
  tests/                    # 12 files, 306 tests
  runs/                     # run output (ignored by git)
```

---

## 19. License

**ISC License** — Copyright (c) 2026 Vincenzo Tilotta.

The full text is in the [LICENSE](LICENSE) file. In short:

| You may | Condition |
|---|---|
| use, copy, modify and distribute the software, for any purpose, with or without fee | the copyright notice and this permission notice must appear in all copies |

The software is provided **"as is"**: no implied warranty of merchantability or fitness, and no liability of the author for any damage arising from its use.

Where the license is declared:

- `LICENSE` — full text;
- `pyproject.toml` — `license = "ISC"` and `license-files = ["LICENSE"]` (SPDX expression, PEP 639): the metadata travels inside the wheel and inside the sdist;
- `zeroai.__license__` — `"ISC"`.

---

**Version**: `0.1.0` (`zeroai.__version__`). **License**: ISC (`zeroai.__license__`), Copyright (c) 2026 Vincenzo Tilotta — see [LICENSE](LICENSE).








