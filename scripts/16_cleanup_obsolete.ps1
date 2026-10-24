param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$taskPrefix = $taskRoot.TrimEnd('\') + '\'
$taskOutput = Join-Path $taskRoot 'outputs\maintenance'

function Get-WorkspaceTarget([string]$Relative) {
    $taskTarget = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $Relative))
    if (-not $taskTarget.StartsWith($taskPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escaped workspace: $Relative"
    }
    $taskCheck = $taskTarget
    while ($taskCheck.StartsWith($taskPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $taskCheck) {
            $taskItem = Get-Item -LiteralPath $taskCheck -Force
            if (($taskItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing cleanup through a reparse point: $taskCheck"
            }
        }
        $taskCheck = Split-Path -Parent $taskCheck
    }
    return $taskTarget
}

# Explicit obsolete entry points; frozen datasets, source evidence and result
# archives are deliberately absent from this allowlist.
$taskDirectories = @(
    'workbench', 'delivery', 'tmp\pdfs', 'outputs_v2',
    'scripts\__pycache__', 'src\uav_project\__pycache__'
)
$taskFiles = @(
    '.refactor_sources.py', 'run_project.py', 'RUN_GUIDE.md', 'START_HERE.ipynb',
    'tools\build_delivery.py', 'configs\default.yaml', 'configs\workflow.json',
    'requirements-colab.txt', 'requirements-v2.txt', 'notebooks\00_colab_setup.ipynb',
    'src\base_config.py', 'src\build_manifest.py', 'src\colab_setup.py',
    'src\crawl_wikimedia.py', 'src\image_features.py', 'src\split_manifest.py',
    'src\train_baselines.py', 'src\train_efficientnet.py',
    'output\pdf\data_audit_report.pdf',
    'outputs\colab_transfer\auxiliary_images.zip',
    'outputs\colab_transfer\COLAB_RUN.ipynb'
)
$taskEmptyDirectories = @(
    'data\public_clean', 'data\public_raw', 'data\self_collected_clean',
    'data\self_collected_raw', 'outputs\exports', 'tmp', 'output\pdf', 'output', 'tools'
)

# The four received preview files can be removed only if their exact bytes are
# already present among preserved raw files. Otherwise keep the evidence.
$taskPreview = Get-WorkspaceTarget 'outputs_v2'
if (Test-Path -LiteralPath $taskPreview) {
    $taskRaw = Get-WorkspaceTarget 'data\raw\public'
    $taskNeeded = @{}
    Get-ChildItem -LiteralPath $taskPreview -Recurse -File | ForEach-Object {
        $taskNeeded[(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash] = $true
    }
    if (Test-Path -LiteralPath $taskRaw) {
        Get-ChildItem -LiteralPath $taskRaw -Recurse -File | ForEach-Object {
            if ($taskNeeded.Count -gt 0) {
                $taskNeeded.Remove((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)
            }
        }
    }
    if ($taskNeeded.Count -gt 0) {
        $taskDirectories = @($taskDirectories | Where-Object { $_ -ne 'outputs_v2' })
        Write-Output 'Preserving outputs_v2: received previews are not proven byte duplicates.'
    }
}

$taskArchive = Get-WorkspaceTarget 'outputs\colab_transfer\auxiliary_images.zip'
if (Test-Path -LiteralPath $taskArchive) {
    $taskMembersPath = Get-WorkspaceTarget 'outputs\colab_transfer\auxiliary_images.zip.members.json'
    $taskMembers = (Get-Content -LiteralPath $taskMembersPath -Raw | ConvertFrom-Json).members
    $taskIndex = 0
    foreach ($taskMember in $taskMembers) {
        $taskOriginal = Get-WorkspaceTarget $taskMember.path
        if (-not (Test-Path -LiteralPath $taskOriginal -PathType Leaf)) {
            throw "Preserved source missing for upload archive: $($taskMember.path)"
        }
        if ((Get-FileHash -LiteralPath $taskOriginal -Algorithm SHA256).Hash -ne $taskMember.sha256) {
            throw "Preserved source differs from upload archive: $($taskMember.path)"
        }
        $taskIndex++
        if ($taskIndex % 2000 -eq 0) { Write-Output "Upload-copy sources verified: $taskIndex / $($taskMembers.Count)" }
    }
    $taskReceipt = Get-Content -LiteralPath (Get-WorkspaceTarget 'outputs\colab_transfer\transfer_receipt.json') -Raw | ConvertFrom-Json
    if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash -ne $taskReceipt.archives.'auxiliary_images.zip'.sha256) {
        throw 'Upload archive differs from its retained receipt'
    }
}
$taskNotebookCopy = Get-WorkspaceTarget 'outputs\colab_transfer\COLAB_RUN.ipynb'
if (Test-Path -LiteralPath $taskNotebookCopy) {
    if ((Get-FileHash -LiteralPath $taskNotebookCopy -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath (Get-WorkspaceTarget 'notebooks\COLAB_RUN.ipynb') -Algorithm SHA256).Hash) {
        throw 'Colab notebook copies differ; preserve both until inspected'
    }
}

$taskTargets = @()
$taskInventory = @()
foreach ($taskRelative in ($taskDirectories + $taskFiles)) {
    $taskAbsolute = Get-WorkspaceTarget $taskRelative
    if (-not (Test-Path -LiteralPath $taskAbsolute)) { continue }
    $taskItem = Get-Item -LiteralPath $taskAbsolute -Force
    $taskTargets += $taskAbsolute
    $taskContents = if ($taskItem.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $taskAbsolute -Recurse -File -Force)
    } else { @($taskItem) }
    foreach ($taskFile in $taskContents) {
        $null = Get-WorkspaceTarget $taskFile.FullName.Substring($taskPrefix.Length)
        $taskInventory += [pscustomobject]@{
            Path=$taskFile.FullName.Substring($taskPrefix.Length)
            Bytes=$taskFile.Length
            SHA256=(Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    if ($taskItem.PSIsContainer) {
        Get-ChildItem -LiteralPath $taskAbsolute -Directory -Recurse -Force | ForEach-Object {
            $null = Get-WorkspaceTarget $_.FullName.Substring($taskPrefix.Length)
        }
    }
}
$taskBytes = ($taskInventory | Measure-Object Bytes -Sum).Sum
Write-Output "Cleanup plan: $($taskInventory.Count) files, $([math]::Round($taskBytes/1MB,2)) MiB."
if (-not $Apply) {
    $taskTargets | Write-Output
    return
}

New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskLedger = Join-Path $taskOutput "cleanup_$taskStamp.csv"
$taskInventory | Export-Csv -LiteralPath $taskLedger -NoTypeInformation -Encoding utf8
foreach ($taskTarget in $taskTargets) {
    # Re-resolve immediately before each recursive deletion.
    $taskChecked = Get-WorkspaceTarget $taskTarget.Substring($taskPrefix.Length)
    Remove-Item -LiteralPath $taskChecked -Recurse -Force
}
foreach ($taskRelative in $taskEmptyDirectories) {
    $taskChecked = Get-WorkspaceTarget $taskRelative
    if ((Test-Path -LiteralPath $taskChecked -PathType Container) -and
        @(Get-ChildItem -LiteralPath $taskChecked -Force).Count -eq 0) {
        Remove-Item -LiteralPath $taskChecked -Force
    }
}
$taskSummary = [ordered]@{
    completed_at=(Get-Date).ToString('o')
    workspace=$taskRoot
    deleted_files=$taskInventory.Count
    deleted_bytes=$taskBytes
    deleted_targets=$taskTargets
    inventory=$taskLedger
    upload_archive_originals_verified=$taskIndex
    preserved=@('data/final/uav_type_v1','data/final/uav_type_v2','data/final/uav_type_v3',
        'data/delivery/uav_dataset_v3','data/raw','data/downloads','outputs/imported_results',
        'outputs/crawl_pages','outputs/platform_inventory','auxiliary_results_for_report.zip','.venv')
}
$taskSummary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOutput "cleanup_$taskStamp.json") -Encoding utf8
$taskSummary | ConvertTo-Json -Depth 5 | Write-Output
