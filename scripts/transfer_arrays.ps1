# Copy prediction arrays from the machine that trained them to a shared transfer folder, one
# TRANSFER.json manifest per configuration. Nothing is deleted or overwritten on the source machine.
#
#   $env:ODDSLIP_ARRAYS_ROOT  = "<local experiments root, e.g. D:\experiments>"
#   $env:ODDSLIP_TRANSFER_DIR = "<transfer folder, e.g. a synced cloud-drive folder>"
#   powershell -ExecutionPolicy Bypass -File scripts\transfer_arrays.ps1
#
# Then, on the receiving machine, run scripts\receive_arrays.py to verify every SHA-256 against the
# canonical hashes recorded in artifacts/results/canonical/*.json and to place the files.
#
# ASCII only on purpose: Windows PowerShell 5.1 reads scripts without a BOM as ANSI.

$ErrorActionPreference = "Stop"
$Thesis = $env:ODDSLIP_ARRAYS_ROOT
$Drive = $env:ODDSLIP_TRANSFER_DIR
if (-not $Thesis) { throw "set ODDSLIP_ARRAYS_ROOT to the experiments root that holds <dataset>\outputs_*" }
if (-not $Drive -or -not (Test-Path $Drive)) { throw "set ODDSLIP_TRANSFER_DIR to an existing transfer folder" }
Write-Host "arrays root:     $Thesis"
Write-Host "transfer folder: $Drive"
$HostName = $env:COMPUTERNAME

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
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    $files = @{}
    Get-ChildItem -LiteralPath $s -File | Where-Object { $_.Name -match '\.(npz|npy|json)$' } | ForEach-Object {
      $h = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
      Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $d $_.Name) -Force
      $files[$_.Name] = @{ size = $_.Length; sha256 = $h }
      Write-Host ("{0,-14} {1,-18} {2,14:N0} B  {3}" -f $j.dataset, $c, $_.Length, $h.Substring(0, 16))
    }
    $manifest = @{ dataset = $j.dataset; config = $c; host = $HostName; files = $files; written = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss") }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $d "TRANSFER.json")
  }
}
Write-Host "done. Wait for the transfer folder to finish syncing before verifying on the other machine."
