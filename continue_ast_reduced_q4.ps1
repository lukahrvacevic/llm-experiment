$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace ".venv\Scripts\python.exe"

Push-Location $workspace
try {
    & $python -m repoexec_baseline.run_matrix `
        --models `
        "deepseek-coder-v2:16b-lite-base-q4_K_M" `
        "codegemma:2b-code-q4_K_M" `
        "codegemma:7b-code-q4_K_M" `
        --subsets full_context `
        --representation ast `
        --task-limit 100 `
        --run-prefix matrix100-pass5-full-ast-q4 `
        --num-return-sequences 5 `
        --do-sample `
        --temperature 0.2 `
        --top-p 0.95 `
        --overwrite

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $python -m repoexec_baseline.run_matrix `
        --subsets full_context `
        --representation reduced_ast `
        --task-limit 100 `
        --run-prefix matrix100-pass5-full-reduced-ast-q4 `
        --num-return-sequences 5 `
        --do-sample `
        --temperature 0.2 `
        --top-p 0.95 `
        --overwrite

    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
