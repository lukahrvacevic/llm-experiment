# RepoExec Raw Baseline

Ovaj workspace sadrzi minimalan baseline za `RepoExec/full_context` bez kompresije konteksta, sa generacijom preko lokalnog Ollama servisa.

Tok je:

1. `repoexec_baseline.generate`
2. `repoexec_baseline.execute`
3. `repoexec_baseline.summarize`

Generacija meri po tasku:

- `input_tokens`
- `output_tokens`
- `generation_seconds`
- `task_wall_seconds`
- `peak_vram_mb`

Summary dodaje:

- `pass@1`
- `DIR`
- zasebne agregate za `cross_context == True`

## Preduslovi

- Python okruzenje sa paketima iz `requirements-baseline.txt`
- kloniran `RepoExec` repo u ovom workspace-u
- pokrenut lokalni Ollama servis
- Docker image za izvrsavanje testova

Instalacija Python paketa:

```powershell
pip install -r requirements-baseline.txt
```

Build Docker image-a za RepoExec execution:

```powershell
docker build -t codeeval-runner -f RepoExec/execution-code-eval/Dockerfile RepoExec/execution-code-eval
```

## 1. Generacija

Primer za jedan model:

```powershell
python -m repoexec_baseline.generate `
  --model deepseek-coder:1.3b `
  --output-dir runs/deepseek-1p3b-full-raw `
  --subset full_context `
  --max-new-tokens 256
```

Za greedy baseline nemoj dodavati `--do-sample`.

Klijent je podrazumevano serijski (`--ollama-parallel-requests 1`). Za pet konkurentnih Pass@5 kandidata pokreni Ollama server sa `OLLAMA_NUM_PARALLEL=5`, pa generaciji prosledi `--ollama-parallel-requests 5`. Broj konkurentnih zahteva ne menja seed ni redosled kandidata.

Primeri modela koje si pominjao:

```powershell
python -m repoexec_baseline.generate --model deepseek-coder:1.3b --output-dir runs/deepseek-1p3b-full-raw
python -m repoexec_baseline.generate --model qwen2.5-coder:3b --output-dir runs/qwen-3b-full-raw
python -m repoexec_baseline.generate --model qwen2.5-coder:7b --output-dir runs/qwen-7b-full-raw
```

Ako Ollama nije na podrazumevanoj adresi, dodaj:

```powershell
--ollama-base-url http://127.0.0.1:11434
```

## 2. Izvrsavanje RepoExec testova

```powershell
python -m repoexec_baseline.execute `
  --repoexec-dir RepoExec `
  --prediction-dir runs/qwen-3b-full-raw
```

Rezultati testova idu u `runs/<ime>/execution/results_<task_id>.jsonl`.

## 3. Zavrsni summary

```powershell
python -m repoexec_baseline.summarize `
  --prediction-dir runs/qwen-3b-full-raw
```

Glavni izlazi su:

- `runs/<ime>/summary.json`
- `runs/<ime>/per_task_metrics.jsonl`
- `runs/<ime>/pre_eval_summary.json`

`summary.json` sadrzi dve sekcije:

- `all_tasks`
- `cross_context_true`

## Napomene

- Ovaj baseline koristi sirovi `prompt` iz RepoExec dataseta, bez AST/call-graph/signature transformacija.
- `pass@1` se racuna iz prve generacije po tasku, sto odgovara pocetnom `num_return_sequences=1` eksperimentu.
- `DIR` se racuna nad prvom generacijom po tasku, pa je direktno uporediv sa tim pocetnim baseline-om.
- `peak_vram_mb` dolazi iz Ollama `/api/ps` kao procena trenutno zauzetog VRAM-a za ucitan model; Ollama ne izbacuje pravi peak po generaciji.

## FMLe generation-only run

`run_fmle_generation.sh` je Linux/SSH launcher za generisanje bez Docker evaluacije. Podrazumevano instalira Ollamu bez `sudo` pristupa u `/workspace/ollama-runtime` i pokrece svih osam modela nad prvih 30 `full_context` taskova, za `raw`, `ast` i `reduced_ast`, sa pet kandidata po tasku.

FMLe launcher i notebook koriste pet paralelnih Ollama zahteva i `num_ctx=4096`. Oni podrazumevano restartuju Ollama server kako bi novo podesavanje bilo primenjeno; to se moze iskljuciti sa `RESTART_OLLAMA=0`. Za serijski rezim, pogodan za lokalni racunar, postavi `OLLAMA_PARALLEL_REQUESTS=1 OLLAMA_NUM_PARALLEL=1`; Python CLI je vec podrazumevano serijski. Promena `OLLAMA_NUM_PARALLEL` zahteva restart Ollama servera.

Za dedicated FMLe Jupyter GPU cvor isti tok je dostupan u `fmle_generation.ipynb`. Notebook proverava Python i GPU, pravi izolovan RepoExec venv bez menjanja CUDA/RAPIDS kernel paketa, koristi user-writable Ollama instalaciju, meri aktivno vreme svake faze i na kraju daje link ka prenosivom generation bundle-u.

Launcher zahteva Python 3.10 ili noviji i vidljiv NVIDIA GPU. Automatski proverava `python3.12`, `python3.11`, `python3.10` i `/opt/conda/bin/python`; alternativno se putanja zadaje kroz `PYTHON_BIN`. Namerno odbija pokretanje na `login01` bez dodeljenog GPU-a.

```bash
chmod +x run_fmle_generation.sh
nohup ./run_fmle_generation.sh > /workspace/repoexec-generation.log 2>&1 &
```

Napredak se prati sa:

```bash
tail -f /workspace/repoexec-generation.log
```

Rezultati i ukupno wall-clock vreme nalaze se u:

```text
/workspace/repoexec-runs/fmle-generation30-generation-summary.json
/workspace/repoexec-runs/fmle-generation30-generation-bundle.tar.gz
```

Ako su modeli vec preuzeti, pre pokretanja se moze postaviti `PULL_MODELS=0`. Prekinut run se ponovo pokrece istom komandom: kompletni run direktorijumi se preskacu, a nepotpuni se generisu ponovo. Nema automatskog retry-ja Ollama zahteva.

Nakon prenosa i raspakivanja bundle-a u lokalni `runs` direktorijum, svi run-ovi se evaluiraju sa:

```powershell
.\.venv\Scripts\python.exe -m repoexec_baseline.evaluate_generation_matrix `
  --matrix-summary runs\fmle-generation30-generation-summary.json `
  --repoexec-dir RepoExec
```
