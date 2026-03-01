param(
    [ValidateSet("dev", "prod")]
    [string]$Mode = "prod",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker command not found. Install Docker Desktop first."
}

try {
    docker compose version | Out-Null
}
catch {
    throw "Docker Compose plugin is not available."
}

$envFile = if ($Mode -eq "prod") { ".env.compose.prod" } else { ".env.compose.dev" }
$overrideFile = if ($Mode -eq "prod") { "docker-compose.prod.yml" } else { "docker-compose.dev.yml" }

if (-not (Test-Path "docker-compose.yml")) {
    throw "docker-compose.yml not found."
}
if (-not (Test-Path $overrideFile)) {
    throw "$overrideFile not found."
}
if (-not (Test-Path $envFile)) {
    $example = if ($Mode -eq "prod") { ".env.compose.prod.example" } else { ".env.compose.dev.example" }
    throw "$envFile not found. Create it from $example first."
}

$apiKeyLine = Get-Content $envFile | Where-Object { $_ -match '^NAMONEXUS_API_KEY=' } | Select-Object -First 1
if (-not $apiKeyLine) {
    throw "NAMONEXUS_API_KEY is missing in $envFile"
}
$apiKey = ($apiKeyLine -split "=", 2)[1].Trim()
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "NAMONEXUS_API_KEY in $envFile is empty"
}

$composeArgs = @("compose", "-f", "docker-compose.yml", "-f", $overrideFile)

Write-Host "[1/4] Building and starting api (mode=$Mode)"
& docker @composeArgs up -d --build api

Write-Host "[2/4] Waiting for health endpoint"
$healthUrl = "http://127.0.0.1:$Port/v1/health"
$ok = $false
for ($i = 0; $i -lt 25; $i++) {
    try {
        $health = Invoke-RestMethod -Method Get -Uri $healthUrl -TimeoutSec 3
        $ok = $true
        break
    }
    catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $ok) {
    & docker @composeArgs logs --tail=200 api
    throw "Health check failed on $healthUrl"
}

Write-Host "[3/4] Calling authenticated update endpoint"
$headers = @{ "X-API-Key" = $apiKey }
$payload = @{
    session_id = "compose_smoke_001"
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
    -Uri "http://127.0.0.1:$Port/v1/fusion/update" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 5

Write-Host "[4/4] Smoke test passed"
Write-Host "`nHealth:"
$health | ConvertTo-Json -Depth 6
Write-Host "`nUpdate:"
$update | ConvertTo-Json -Depth 8

Write-Host "`nTo stop:"
Write-Host "docker compose -f docker-compose.yml -f $overrideFile down"
