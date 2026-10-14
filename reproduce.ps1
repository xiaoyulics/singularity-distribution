# Regenerate every number, check and figure of the paper (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
foreach ($step in @("make_numbers.py", "run_checks.py", "check_small_time.py", "make_figures.py --recompute", "make_hero.py", "make_hero_labels.py", "make_web_data.py")) {
    $parts = $step.Split(" ")
    python @parts
    if ($LASTEXITCODE -ne 0) { throw "$step failed" }
}
