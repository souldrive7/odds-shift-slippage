# Copy prediction arrays from the machine that trained them (EB, the ASUS ExpertBook "mothership") to
# the Google Drive transfer folder, one TRANSFER.json manifest per configuration (same format the AW
# machine used on 2026-09-08). Run on EB in PowerShell. Nothing is deleted or overwritten on EB.
#
#   powershell -ExecutionPolicy Bypass -File scripts\transfer_arrays.ps1
#
# Then, on the receiving machine, run scripts\receive_arrays.py to verify every SHA-256 against the
# canonical hashes recorded in artifacts/results/canonical/*.json and to place the files.

$ErrorActionPreference = "Stop"
$Thesis = "C:\dev\msc-thesis\experiments"
$Drive  = "G:\マイドライブ\06_研究\00_msc_Thesis\transfer"
$Host_  = "EB"

$jobs = @(
  @{ dataset = "santander"; src = "$Thesis\santander_product_proxy\outputs_matched_pair"; dst = "$Drive\santander_outputs_matched_pair";
     configs = @("unweighted", "weighted", "weighted_mds0.7", "weighted_mds2") },
  @{ dataset = "instacart"; src = "$Thesis\instacart_product\outputs_matched_pair"; dst = "$Drive\outputs_matched_pair";
     configs = @("unweighted", "weighted", "weighted_mds0.3", "weighted_mds0.7", "weighted_mds1", "weighted_mds2", "weighted_mds5") },
  @{ dataset = "santander_mlp"; src = "$Thesis\santander_product_proxy\outputs_mlp"; dst = "$Drive\santander_outputs_mlp";
     configs = @("w1", "wr") },
  @{ dataset = "instacart_mlp"; src = "$Thesis\instacart_product\outputs_mlp"; dst = "$Drive\instacart_outputs_mlp";
     configs = @("w1", "wr") }
)

foreach ($j in $jobs) {
  foreach ($c in $j.configs) {
    $s = Join-Path $j.src $c
    if (-not (Test-Path $s)) { Write-Host "SKIP (not on this machine): $s"; continue }
    $d = Join-Path $j.dst $c
    New-Item -ItemType Directory -Force $d | Out-Null
    $files = @{}
    Get-ChildItem $s -File | Where-Object { $_.Name -match '\.(npz|npy|json)$' } | ForEach-Object {
      $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
      Copy-Item $_.FullName (Join-Path $d $_.Name) -Force
      $files[$_.Name] = @{ size = $_.Length; sha256 = $h }
      Write-Host ("{0,-14} {1,-18} {2,12:N0} B  {3}" -f $j.dataset, $c, $_.Length, $h.Substring(0, 16))
    }
    $manifest = @{ dataset = $j.dataset; config = $c; host = $Host_; files = $files; written = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $d "TRANSFER.json")
  }
}
Write-Host "done. Wait for Google Drive to finish uploading before verifying on the other machine."
