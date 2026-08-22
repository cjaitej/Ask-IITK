<#
.SYNOPSIS
    Build, push and deploy AskIITK to Azure Container Apps.

.DESCRIPTION
    Idempotent. Re-running pushes a new tag and updates the app in place.

    The Azure and Docker Hub names are iitk-rag, from before the project was
    called AskIITK. Neither can be renamed in place, and both are only ever
    seen in the URL, so they stay as they are.

    Public Docker Hub image, own resource group, shared Container Apps
    environment — the same shape as the other apps here. No ACR, so nothing
    bills between deploys.

    The image is self-contained: model and index are baked in, so the app
    needs no sidecar or volume. GEMINI_API_KEY is injected as a secret.

.EXAMPLE
    .\deploy\azure.ps1
    .\deploy\azure.ps1 -Tag v2 -MinReplicas 1
    .\deploy\azure.ps1 -SkipBuild            # push + deploy an image already built
#>
[CmdletBinding()]
param(
    [string] $ResourceGroup = "iitk-rag-rg",
    [string] $Location      = "centralindia",
    [string] $AppName       = "iitk-rag",

    # The shared environment, which lives in another resource group. An app and
    # its environment need not be co-located; diffusion-app is placed the same way.
    [string] $EnvName          = "coregpt-env",
    [string] $EnvResourceGroup = "coregpt-rg",

    # Public repo. The image carries the corpus and index but no secrets:
    # .dockerignore excludes .env, and the Gemini key is set as a secret below.
    [string] $DockerHubUser = "cjaitej",
    [string] $Repository    = "iitk-rag",
    [string] $Tag           = "v1",

    # 0 = scale to zero (costs nothing idle, ~30s cold start while torch loads).
    # 1 = keep one replica warm.
    [int]    $MinReplicas   = 0,
    [int]    $MaxReplicas   = 2,
    [string] $Cpu           = "1.0",
    [string] $Memory        = "2.0Gi",

    # Defaults to GEMINI_API_KEY / GEMINI_MODEL from .env, then the environment.
    [string] $GeminiApiKey  = "",
    [string] $GeminiModel   = "",

    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$image    = "$DockerHubUser/$Repository" + ":" + "$Tag"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# Windows PowerShell turns a native command's redirected stderr into terminating
# errors under $ErrorActionPreference = "Stop", and az writes a routine warning
# there on every containerapp call. Piping this script's output was enough to
# abort a deploy halfway through. So native calls go through here, where the
# exit code decides.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)] [string] $Exe,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [string] $What = ""
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) {
            $label = if ($What) { $What } else { "$Exe $($Arguments -join ' ')" }
            throw "$label failed (exit $LASTEXITCODE)."
        }
        return $output
    }
    finally { $ErrorActionPreference = $prev }
}

function Invoke-Az {
    param([Parameter(Mandatory = $true)] [string[]] $AzArgs, [string] $What = "")
    return Invoke-Native -Exe "az" -Arguments $AzArgs -What $What
}

# "Does this exist?" — a non-zero exit is the answer, not a failure.
function Test-AzResource {
    param([Parameter(Mandatory = $true)] [string[]] $AzArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & az @AzArgs --output none 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    finally { $ErrorActionPreference = $prev }
}

# Reads one KEY=value out of .env without sourcing the file.
function Get-EnvValue {
    param([string] $Key)
    $envFile = Join-Path $repoRoot ".env"
    if (-not (Test-Path $envFile)) { return "" }
    $match = Select-String -Path $envFile -Pattern "^\s*$Key\s*=\s*(.+)$"
    if ($match) { return $match.Matches[0].Groups[1].Value.Trim() }
    return ""
}

# --- 0. prerequisites ------------------------------------------------------
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI not found. Install it: https://aka.ms/installazurecli"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker not found. It builds and pushes the image."
}
if (-not (Test-AzResource @("account", "show"))) {
    throw "Not logged in to Azure. Run: az login"
}
$sub = (Invoke-Az @("account", "show", "--output", "json") | ConvertFrom-Json)
Write-Host "Subscription: $($sub.name)  ($($sub.id))"

Step "Azure CLI extensions"
# Explicit, so a first run does not stall on the dynamic-install prompt.
Invoke-Az @("extension", "add", "--name", "containerapp", "--upgrade", "--only-show-errors") -What "az extension add containerapp" | Out-Null

# Read but never printed, and never baked into the image.
if (-not $GeminiApiKey) { $GeminiApiKey = Get-EnvValue "GEMINI_API_KEY" }
if (-not $GeminiApiKey) { $GeminiApiKey = $env:GEMINI_API_KEY }
if (-not $GeminiApiKey) {
    throw "No GEMINI_API_KEY found. Pass -GeminiApiKey, or set it in .env. Without it retrieval still works but /chat returns 503."
}
if (-not $GeminiModel) { $GeminiModel = Get-EnvValue "GEMINI_MODEL" }
if (-not $GeminiModel) { $GeminiModel = "gemini-2.5-flash" }
Write-Host "Gemini model: $GeminiModel"

# --- 1. build --------------------------------------------------------------
if ($SkipBuild) {
    Step "Skipping build (-SkipBuild)"
}
else {
    Step "Building $image (several minutes on a cold build)"
    # Container Apps runs amd64; pinned so an ARM machine cannot build an image
    # that will not start there.
    Invoke-Native -Exe "docker" -What "docker build" -Arguments @(
        "build", "--platform", "linux/amd64", "-t", $image, $repoRoot)
}

# --- 2. push ---------------------------------------------------------------
Step "Pushing $image to Docker Hub"
try {
    Invoke-Native -Exe "docker" -What "docker push" -Arguments @("push", $image)
}
catch {
    throw "docker push failed. Run 'docker login' first, and make sure the repo $DockerHubUser/$Repository is yours. ($_)"
}

# --- 3. resource group -----------------------------------------------------
Step "Resource group: $ResourceGroup"
Invoke-Az @("group", "create", "--name", $ResourceGroup, "--location", $Location, "--output", "none") | Out-Null

# --- 4. the shared environment --------------------------------------------
Step "Container Apps environment: $EnvName (in $EnvResourceGroup)"
if (-not (Test-AzResource @("containerapp", "env", "show", "--name", $EnvName, "--resource-group", $EnvResourceGroup))) {
    throw "Environment '$EnvName' not found in resource group '$EnvResourceGroup'. Pass -EnvName/-EnvResourceGroup, or create one with: az containerapp env create -n $EnvName -g $EnvResourceGroup -l $Location"
}
# By resource id, which is what lets the app live in a different group.
$envId = Invoke-Az @("containerapp", "env", "show", "--name", $EnvName,
    "--resource-group", $EnvResourceGroup, "--query", "id", "--output", "tsv")

# --- 5. the app ------------------------------------------------------------
$envVars = @(
    "GEMINI_API_KEY=secretref:gemini-api-key",
    "GEMINI_MODEL=$GeminiModel",
    "QDRANT_MODE=local",
    "TOP_K=6"
)

if (-not (Test-AzResource @("containerapp", "show", "--name", $AppName, "--resource-group", $ResourceGroup))) {
    Step "Creating container app: $AppName"
    Invoke-Az (@(
        "containerapp", "create",
        "--name", $AppName,
        "--resource-group", $ResourceGroup,
        "--environment", $envId,
        "--image", $image,
        "--target-port", "8000",
        "--ingress", "external",
        "--cpu", $Cpu, "--memory", $Memory,
        "--min-replicas", $MinReplicas, "--max-replicas", $MaxReplicas,
        "--secrets", "gemini-api-key=$GeminiApiKey",
        "--env-vars") + $envVars + @("--output", "none")) -What "containerapp create" | Out-Null
}
else {
    Step "Updating container app: $AppName"
    Invoke-Az @("containerapp", "secret", "set", "--name", $AppName,
        "--resource-group", $ResourceGroup,
        "--secrets", "gemini-api-key=$GeminiApiKey", "--output", "none") | Out-Null
    Invoke-Az (@(
        "containerapp", "update",
        "--name", $AppName,
        "--resource-group", $ResourceGroup,
        "--image", $image,
        "--min-replicas", $MinReplicas, "--max-replicas", $MaxReplicas,
        "--set-env-vars") + $envVars + @("--output", "none")) -What "containerapp update" | Out-Null
}

# A freshly pushed tag can take a moment to become pullable.
Step "Waiting for the revision to come up"
$fqdn = Invoke-Az @("containerapp", "show", "--name", $AppName,
    "--resource-group", $ResourceGroup,
    "--query", "properties.configuration.ingress.fqdn", "--output", "tsv")
$state = Invoke-Az @("containerapp", "revision", "list", "--name", $AppName,
    "--resource-group", $ResourceGroup,
    "--query", "[-1].properties.runningState", "--output", "tsv")
Write-Host "  latest revision: $state"

# --- 6. done ---------------------------------------------------------------
Step "Deployed"
Write-Host "  UI      https://$fqdn/ui"
Write-Host "  Docs    https://$fqdn/docs"
Write-Host "  Health  https://$fqdn/health"
Write-Host ""
Write-Host "First request after an idle period pays a cold start while the model loads."
Write-Host "Logs:   az containerapp logs show -n $AppName -g $ResourceGroup --follow"
Write-Host "Delete: az group delete -n $ResourceGroup --yes --no-wait"
Write-Host ""
Write-Host "Note: deleting $ResourceGroup leaves $EnvName alone — it lives in $EnvResourceGroup"
Write-Host "and is shared with your other apps."
