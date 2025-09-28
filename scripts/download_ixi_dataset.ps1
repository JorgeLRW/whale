param(
    [Parameter()]
    [string]$OutputRoot,

    [string[]]$Archives,

    [switch]$SkipExtract,

    [switch]$SkipExtras,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $scriptRoot "..\data\IXI"
}

function Write-Section {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Invoke-ResumableDownload {
    param(
        [string]$Uri,
        [string]$Destination
    )

    if ((Test-Path $Destination -PathType Leaf) -and -not $Force) {
        Write-Host "Already present: $Destination" -ForegroundColor Yellow
        return
    }

    $tmpDestination = "$Destination.partial"
    if (Test-Path $tmpDestination) {
        Remove-Item $tmpDestination -Force
    }

    try {
        Invoke-WebRequest -Uri $Uri -OutFile $tmpDestination -UseBasicParsing
    }
    catch {
        Write-Warning "Invoke-WebRequest failed for $Uri";
        Write-Warning $_.Exception.Message
        Write-Host "Falling back to BITS transfer..."
        Start-BitsTransfer -Source $Uri -Destination $tmpDestination -DisplayName "IXI download" -TransferType Download
    }

    Move-Item -Path $tmpDestination -Destination $Destination -Force
}

function Test-TarAvailable {
    if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
        throw "The 'tar' utility is required but was not found. Install 7-Zip or add GNU tar to PATH, then re-run."
    }
}

$archiveMap = @{
    "IXI-T1.tar" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar";
    "IXI-T2.tar" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar";
    "IXI-PD.tar" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar";
    "IXI-MRA.tar" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar";
    "IXI-DTI.tar" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar"
}

$extras = @{
    "bvecs.txt" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/bvecs.txt";
    "bvals.txt" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/bvals.txt";
    "IXI.xls" = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
}

$resolvedArchives = @()
if (-not $Archives -or $Archives.Count -eq 0) {
    $resolvedArchives = $archiveMap.Keys
}
else {
    foreach ($item in $Archives) {
        if (-not $item) { continue }
        $token = $item.Trim()
        if (-not $token) { continue }
        if (-not $token.EndsWith('.tar', [System.StringComparison]::OrdinalIgnoreCase)) {
            if (-not $token.StartsWith('IXI-', [System.StringComparison]::OrdinalIgnoreCase)) {
                $token = "IXI-$token"
            }
            $token = "$token.tar"
        }
        $key = $archiveMap.Keys | Where-Object { $_.Equals($token, 'OrdinalIgnoreCase') }
        if (-not $key) {
            throw "Requested archive '$item' not recognized. Available archives: $($archiveMap.Keys -join ', ')"
        }
        $resolvedArchives += $key
    }
}
$resolvedArchives = $resolvedArchives | Sort-Object -Unique

$OutputRoot = (Resolve-Path (New-Item -ItemType Directory -Path $OutputRoot -Force)).Path
Write-Section "Target directory"
Write-Host $OutputRoot

Write-Section "Downloading archives"
foreach ($name in $resolvedArchives) {
    $uri = $archiveMap[$name]
    if (-not $uri) {
        Write-Warning "No URI found for archive '$name'; skipping."
        continue
    }
    $fileName = $name
    $destination = Join-Path $OutputRoot $fileName
    Write-Host "Fetching $fileName ..."
    Invoke-ResumableDownload -Uri $uri -Destination $destination

    if (-not $SkipExtract) {
        Test-TarAvailable
        $extractMarker = Join-Path $OutputRoot ("." + $fileName + ".extracted")
        if ((Test-Path $extractMarker) -and -not $Force) {
            Write-Host "Already extracted: $fileName" -ForegroundColor Yellow
            continue
        }

        Write-Host "Extracting $fileName ..."
        tar -xf $destination -C $OutputRoot
        New-Item -ItemType File -Path $extractMarker -Force | Out-Null
    }
}

if (-not $SkipExtras) {
    Write-Section "Downloading auxiliary files"
    foreach ($item in $extras.GetEnumerator()) {
        $fileName = $item.Key
        $uri = $item.Value
        $destination = Join-Path $OutputRoot $fileName
        Write-Host "Fetching $fileName ..."
        Invoke-ResumableDownload -Uri $uri -Destination $destination
    }
}

Write-Host "`nAll requested IXI assets are present in $OutputRoot" -ForegroundColor Green
