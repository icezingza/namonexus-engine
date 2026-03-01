param(
    [string]$ImageTag = "namonexus-fusion:prod-smoke",
    [string]$ContainerName = "namonexus-fusion-smoke",
    [string]$ApiKey = "replace-with-strong-api-key",
    [int]$HostPort = 8000
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker command not found. Install Docker Desktop first."
}

Write-Host "[1/5] Building image $ImageTag"
docker build -t $ImageTag .

Write-Host "[2/5] Removing old container (if exists)"
docker rm -f $ContainerName 2>$null | Out-Null

Write-Host "[3/5] Starting container $ContainerName on port $HostPort"
docker run -d `
  --name $ContainerName `
  -p "${HostPort}:8000" `
  -e "NAMONEXUS_API_KEY=$ApiKey" `
  -e "NAMONEXUS_ALLOWED_ORIGINS=https://app.example.com" `
  -e "NAMONEXUS_LAWFUL_BASIS=contract" `
  $ImageTag | Out-Null

Write-Host "[4/5] Waiting for health endpoint"
$healthUrl = "http://127.0.0.1:$HostPort/v1/health"
$ok = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $health = Invoke-RestMethod -Method Get -Uri $healthUrl
        $ok = $true
        break
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ok) {
    Write-Host "Container logs:"
    docker logs $ContainerName
    throw "Health check failed"
}

Write-Host "[5/5] Running authenticated smoke update"
$headers = @{ "X-API-Key" = $ApiKey }
$payload = @{
    session_id = "docker_smoke_001"
    score = 0.72
    confidence = 0.88
    modality = "text"
    metadata = @{
        device = "mobile"
        auth_token = "should_be_dropped"
    }
} | ConvertTo-Json -Depth 6

$update = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:$HostPort/v1/fusion/update" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload

Write-Host "`nHealth:"
$health | ConvertTo-Json -Depth 6
Write-Host "`nUpdate:"
$update | ConvertTo-Json -Depth 8

Write-Host "`nSmoke test passed. Container is running: $ContainerName"
Write-Host "Stop it with: docker rm -f $ContainerName"
