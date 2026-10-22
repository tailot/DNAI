<#
.SYNOPSIS
    ZeroAI - demo end-to-end con modelli reali (famiglia Qwen2.5-0.5B).

.DESCRIPTION
    Sequenza completa:
      1. doctor   : stato di python / torch / CUDA / mergekit
      2. download : base + 2 expert della stessa famiglia (~3 GB di download)
      3. merge    : un solo merge, nessuna ottimizzazione (prova minima del percorso reale)
      4. search   : ricerca evolutiva su modelli reali (budget basso)
      5. results  : summary.md, top candidati, genoma migliore, best_mergekit.yaml, test di generazione

    Ambiente richiesto:
        pip install -r requirements.txt
        pip install -e .

    Attenzione al disco: in --mode model l'opzione --write-checkpoints e obbligatoria
    (LLMEvaluator valuta solo checkpoint scritti su disco) e ogni candidato occupa ~1 GB
    in runs/<run-name>/candidates/. Il budget predefinito di questo script e 4 candidati
    (population 4 x generations 1, oppure 4 Optuna trials) quindi ~4 GB.

    Nota: questo file e volutamente senza accenti per essere leggibile anche da
    Windows PowerShell 5.1 (che interpreta i .ps1 senza BOM come ANSI/CP1252).

.PARAMETER Step
    doctor | download | merge | search | results | all   (default: all)

.EXAMPLE
    .\scripts\real_model_demo.ps1

.EXAMPLE
    .\scripts\real_model_demo.ps1 -Step search -Optimizer optuna -Trials 8 -Blocks 4

.EXAMPLE
    .\scripts\real_model_demo.ps1 -Step merge -Device cpu -Dtype float32

.EXAMPLE
    .\scripts\real_model_demo.ps1 -Step results -SkipGenerationTest
#>
[CmdletBinding()]
param(
    [ValidateSet("doctor", "download", "merge", "search", "results", "all")]
    [string]$Step = "all",

    [string]$PythonExe = "python",
    [string]$ModelDir = "models",

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [ValidateSet("bfloat16", "float16", "float32")]
    [string]$Dtype = "bfloat16",

    [int]$Blocks = 4,

    [ValidateSet("optuna", "genetic", "none")]
    [string]$Optimizer = "genetic",

    [int]$Trials = 6,
    [int]$Population = 4,
    [int]$Generations = 1,
    [int]$Seed = 1337,

    [string]$MergeRunName = "demo_merge",
    [string]$SearchRunName = "demo_search",

    [switch]$SkipGenerationTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# The script lives in <repo>/scripts: every path below is relative to the repo root.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $RepoRoot

$Base = Join-Path $ModelDir "base"
$Instruct = Join-Path $ModelDir "instruct"
$Coder = Join-Path $ModelDir "coder"

function Write-Title([string]$text) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
}

function Invoke-Python {
    param([string]$StepName, [string[]]$Arguments)
    Write-Host ""
    Write-Host ("$ {0} {1}" -f $PythonExe, ($Arguments -join " ")) -ForegroundColor DarkGray
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "step '$StepName' fallita: exit code $LASTEXITCODE"
    }
}

function Require-Models {
    $missing = @($Base, $Instruct, $Coder) |
        Where-Object { -not (Test-Path (Join-Path $_ "model.safetensors")) }
    if ($missing) {
        throw ("checkpoint mancanti: {0}. Esegui prima: .\scripts\real_model_demo.ps1 -Step download" -f ($missing -join ", "))
    }
}

# ------------------------------------------------------------------- 1. doctor
function Step-Doctor {
    Write-Title "1/5  doctor - python, torch, CUDA, mergekit"
    Invoke-Python "doctor" @("-m", "zeroai.cli", "doctor")
}

# ----------------------------------------------------------------- 2. download
function Step-Download {
    Write-Title "2/5  download dei 3 checkpoint Qwen2.5-0.5B (~3 GB)"
    Invoke-Python "download" @("scripts/download_demo_models.py", "--dir", $ModelDir)
}

# -------------------------------------------------------------------- 3. merge
function Step-Merge {
    Require-Models
    Write-Title "3/5  un solo merge (optimizer=none) - prova minima del percorso reale"
    # merge = run con optimizer=none, write_checkpoints=True, merge_backend=native.
    # --blocks 2 con 2 modelli: --weights accetta 4 valori in ordine model-major
    # instruct = [0.2, 0.8], coder = [0.8, 0.2]
    Invoke-Python "merge" @(
        "-m", "zeroai.cli", "merge",
        "--mode", "model",
        "--base", $Base,
        "--models", $Instruct, $Coder,
        "--method", "dare_ties",
        "--weights", "0.2", "0.8", "0.8", "0.2",
        "--dare-density", "0.4",
        "--blocks", "2",
        "--dtype", $Dtype,
        "--device", $Device,
        "--run-name", $MergeRunName,
        "--log-level", "WARNING"
    )
}

# ------------------------------------------------------------------- 4. search
function Step-Search {
    Require-Models
    Write-Title "4/5  ricerca evolutiva su modelli reali ($Optimizer)"
    # GeneticOptimizer requires population >= 4 and generations >= 1.
    # Candidates written to disk = population * generations (or trials for Optuna).
    Invoke-Python "search" @(
        "-m", "zeroai.cli", "run",
        "--mode", "model",
        "--base", $Base,
        "--models", $Instruct, $Coder,
        "--blocks", "$Blocks",
        "--dtype", $Dtype,
        "--device", $Device,
        "--seed", "$Seed",
        "--optimizer", $Optimizer,
        "--trials", "$Trials",
        "--population", "$Population",
        "--generations", "$Generations",
        "--methods", "linear", "ties", "dare_linear", "dare_ties",
        "--weight-low", "0.0",
        "--weight-high", "1.2",
        "--merge-backend", "native",
        "--write-checkpoints",
        "--run-name", $SearchRunName,
        "--log-level", "INFO"
    )
}

# ------------------------------------------------------------------ 5. results
function Step-Results {
    Write-Title "5/5  lettura dei risultati"

    $runDir = Join-Path "runs" $SearchRunName
    if (-not (Test-Path (Join-Path $runDir "summary.json"))) {
        $runDir = Join-Path "runs" $MergeRunName
    }
    if (-not (Test-Path (Join-Path $runDir "summary.json"))) {
        throw "nessuna run trovata in runs\$SearchRunName o runs\$MergeRunName"
    }

    Write-Host ""
    Write-Host "--- $runDir\summary.md ---" -ForegroundColor Yellow
    Get-Content (Join-Path $runDir "summary.md")

    Write-Host ""
    Write-Host "--- top 5 candidati (candidates.csv) ---" -ForegroundColor Yellow
    Import-Csv (Join-Path $runDir "candidates.csv") |
        Sort-Object { [double]$_.fitness } -Descending |
        Select-Object -First 5 candidate_id, method, fitness, density, dare_density, eval_seconds |
        Format-Table -AutoSize

    $summary = Get-Content (Join-Path $runDir "summary.json") -Raw | ConvertFrom-Json
    $scored = @($summary.candidates | Where-Object { -not $_.error } | Sort-Object fitness -Descending)
    if (-not $scored) {
        Write-Warning "nessun candidato valutato con successo: controlla la colonna error in candidates.csv"
        return
    }
    $best = $scored[0]

    Write-Host ""
    Write-Host "--- genoma e metriche del candidato migliore ---" -ForegroundColor Yellow
    $best | ConvertTo-Json -Depth 6

    $yamlPath = Join-Path $runDir "best_mergekit.yaml"
    if (Test-Path $yamlPath) {
        Write-Host ""
        Write-Host "--- best_mergekit.yaml (pronto per mergekit-run) ---" -ForegroundColor Yellow
        Get-Content $yamlPath
    }

    $bestDir = Join-Path (Join-Path $runDir "candidates") $best.candidate_id
    if ((Test-Path $bestDir) -and -not $SkipGenerationTest) {
        Write-Host ""
        Write-Host "--- generazione di prova dal checkpoint fuso ($bestDir) ---" -ForegroundColor Yellow
        Invoke-Python "generation-test" @(
            "scripts/test_merged_model.py", $bestDir,
            "--device", $Device,
            "--dtype", $Dtype,
            "--max-new-tokens", "200"
        )
    }

    $checkpoints = @(Get-ChildItem $runDir -Recurse -File | Where-Object { $_.Extension -eq ".safetensors" })
    Write-Host ""
    if ($checkpoints.Count -eq 0) {
        Write-Host ("nessun checkpoint .safetensors in {0}" -f $runDir) -ForegroundColor DarkGray
    } else {
        $bytes = 0
        foreach ($file in $checkpoints) { $bytes += $file.Length }
        Write-Host ("{0} checkpoint scritti in {1}: {2:N2} GB" -f $checkpoints.Count, $runDir, ($bytes / 1GB)) -ForegroundColor DarkGray
    }
}

# --------------------------------------------------------------------- dispatch
if (-not (Get-Command $PythonExe -ErrorAction SilentlyContinue)) {
    throw "interprete '$PythonExe' non trovato: passa -PythonExe con il percorso del tuo ambiente"
}

switch ($Step) {
    "doctor"   { Step-Doctor }
    "download" { Step-Download }
    "merge"    { Step-Merge }
    "search"   { Step-Search }
    "results"  { Step-Results }
    "all"      {
        Step-Doctor
        Step-Download
        Step-Merge
        Step-Search
        Step-Results
    }
}