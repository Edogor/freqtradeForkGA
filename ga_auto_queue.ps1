<#
.SYNOPSIS
    GA Auto-Queue Daemon (PowerShell) â€” Keep the laptop busy with GA experiments 24/7

.DESCRIPTION
    PowerShell port of ga_auto_queue_v2.sh. Monitors running GA processes and
    launches queued experiments to maintain a target number of concurrent runs.

    Queue directory:  genetic_algorithm/config/queue/
      - YAML configs named with priority prefix: 01_V1_name.yaml
      - Lower number = higher priority (launched first)

    Done directory:   genetic_algorithm/config/done/{wave}/
      - Completed configs organized by wave

.PARAMETER Wave
    Wave name (default: auto-detect next wave)

.PARAMETER Max
    Max concurrent experiments (default: 5)

.PARAMETER Persistent
    Keep running after queue drains (watch for new configs)

.PARAMETER Poll
    Poll interval in seconds (default: 30)

.PARAMETER Status
    Show queue status and exit

.PARAMETER Stop
    Stop daemon gracefully

.EXAMPLE
    .\ga_auto_queue.ps1 -Wave wave26_laptop -Max 5 -Persistent
    .\ga_auto_queue.ps1 -Status
    .\ga_auto_queue.ps1 -Stop
#>

[CmdletBinding(DefaultParameterSetName = 'Run')]
param(
    [Parameter(ParameterSetName = 'Run')]
    [string]$Wave = "",

    [Parameter(ParameterSetName = 'Run')]
    [int]$Max = 5,

    [Parameter(ParameterSetName = 'Run')]
    [switch]$Persistent,

    [Parameter(ParameterSetName = 'Run')]
    [int]$Poll = 30,

    [Parameter(ParameterSetName = 'Status')]
    [switch]$Status,

    [Parameter(ParameterSetName = 'Stop')]
    [switch]$Stop
)

# â”€â”€ Paths â”€â”€
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$QueueDir = Join-Path $RepoDir "genetic_algorithm\config\queue"
$DoneBase = Join-Path $RepoDir "genetic_algorithm\config\done"
$LogDir = Join-Path $RepoDir "genetic_algorithm\logs"
$OutputBase = Join-Path $RepoDir "genetic_algorithm\output\exploration"
$LogFile = Join-Path $LogDir "auto_queue.log"
$PidFile = Join-Path $LogDir "auto_queue_daemon.pid"
$TrackedFile = Join-Path $LogDir "auto_queue_tracked.txt"
$StateFile = Join-Path $LogDir "auto_queue_state.json"

# Ensure directories exist
@($QueueDir, $DoneBase, $LogDir) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

# â”€â”€ Logging â”€â”€
function Write-Log {
    param([string]$Level, [string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts [$Level] $Message"
    Add-Content -Path $LogFile -Value $line -ErrorAction SilentlyContinue
    if ($Level -eq "ERROR") {
        Write-Host "[$ts] $Message" -ForegroundColor Red
    } elseif ($Level -eq "WARNING") {
        Write-Host "[$ts] $Message" -ForegroundColor Yellow
    } else {
        Write-Host "[$ts] $Message" -ForegroundColor Cyan
    }
}

# â”€â”€ Helper Functions â”€â”€
function Get-QueuedConfigs {
    Get-ChildItem -Path $QueueDir -Filter "*.yaml" -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -ExpandProperty Name
}

function Get-ExperimentName {
    param([string]$ConfigBasename)
    $name = $ConfigBasename -replace '\.yaml$', ''
    $name = $name -replace '^\d+_', ''
    return $name
}

function Get-RunningGAProcesses {
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*run_ga.py*" } |
        Measure-Object |
        Select-Object -ExpandProperty Count
}

function Save-State {
    param(
        [string]$WaveName,
        [int]$MaxConcurrent,
        [bool]$IsPersistent,
        [int]$LaunchedCount,
        [int]$CompletedCount,
        [int]$FailedCount
    )
    $state = @{
        wave = $WaveName
        max_concurrent = $MaxConcurrent
        persistent = $IsPersistent
        daemon_pid = $PID
        started_at = (Get-Date -Format "o")
        updated_at = (Get-Date -Format "o")
        launched_count = $LaunchedCount
        completed_count = $CompletedCount
        failed_count = $FailedCount
    }
    $state | ConvertTo-Json | Set-Content -Path $StateFile -ErrorAction SilentlyContinue
}

# â”€â”€ Status Command â”€â”€
if ($Status) {
    $running = Get-RunningGAProcesses
    $queued = (Get-QueuedConfigs).Count
    $doneCount = (Get-ChildItem -Path $DoneBase -Filter "*.yaml" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count

    Write-Host ""
    Write-Host "  +====================================+" -ForegroundColor Cyan
    Write-Host "  |   GA Auto-Queue Status             |" -ForegroundColor Cyan
    Write-Host "  +====================================+" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Running:    $running / $Max" -ForegroundColor Green
    Write-Host "  Queued:     $queued" -ForegroundColor Yellow
    Write-Host "  Completed:  $doneCount" -ForegroundColor DarkGray

    if (Test-Path $PidFile) {
        $daemonPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        try {
            $proc = Get-Process -Id $daemonPid -ErrorAction Stop
            Write-Host "  * Daemon running (PID $daemonPid)" -ForegroundColor Green
        } catch {
            Write-Host "  o Daemon not running (stale PID file)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  o Daemon not running" -ForegroundColor Yellow
    }

    if (Test-Path $StateFile) {
        $stateData = Get-Content $StateFile | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($stateData) {
            Write-Host "  Wave:       $($stateData.wave)" -ForegroundColor White
        }
    }

    $configs = Get-QueuedConfigs
    if ($configs) {
        Write-Host ""
        Write-Host "  Pending queue:" -ForegroundColor White
        foreach ($c in $configs) {
            Write-Host "    -> $($c -replace '\.yaml$', '')" -ForegroundColor DarkGray
        }
    }

    # Show tracked experiments
    if (Test-Path $TrackedFile) {
        $tracked = Get-Content $TrackedFile -ErrorAction SilentlyContinue |
            Where-Object { $_ -notmatch '^\s*#' -and $_ -match '\S' }
        if ($tracked) {
            Write-Host ""
            Write-Host "  Active experiments:" -ForegroundColor White
            foreach ($line in $tracked) {
                $parts = $line -split '\s+', 2
                if ($parts.Count -ge 2) {
                    $pid = $parts[0]
                    $name = $parts[1]
                    try {
                        $proc = Get-Process -Id $pid -ErrorAction Stop
                        Write-Host "    * $name (PID $pid)" -ForegroundColor Green
                    } catch {
                        Write-Host "    o $name (PID $pid, finished)" -ForegroundColor DarkGray
                    }
                }
            }
        }
    }
    Write-Host ""
    exit 0
}

# â”€â”€ Stop Command â”€â”€
if ($Stop) {
    if (Test-Path $PidFile) {
        $daemonPid = Get-Content $PidFile -ErrorAction SilentlyContinue
        try {
            $proc = Get-Process -Id $daemonPid -ErrorAction Stop
            Write-Host "Sending stop signal to daemon (PID $daemonPid)..." -ForegroundColor Yellow
            Stop-Process -Id $daemonPid -Force
            Write-Host "Daemon stopped. Running experiments will continue to completion." -ForegroundColor Green
        } catch {
            Write-Host "Daemon not running (stale PID file). Cleaning up." -ForegroundColor Yellow
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "No daemon PID file found." -ForegroundColor Yellow
    }
    exit 0
}

# â”€â”€ Pre-flight Checks â”€â”€
if (Test-Path $PidFile) {
    $existingPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    try {
        $proc = Get-Process -Id $existingPid -ErrorAction Stop
        Write-Host "ERROR: Daemon already running (PID $existingPid). Use -Stop first." -ForegroundColor Red
        exit 1
    } catch {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

# â”€â”€ Determine Wave Name â”€â”€
if (-not $Wave) {
    $existingWaves = Get-ChildItem -Path $OutputBase -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^wave(\d+)' } |
        ForEach-Object { [int]($_.Name -replace '\D', '') } |
        Sort-Object -Descending
    if ($existingWaves) {
        $nextNum = $existingWaves[0] + 1
        $Wave = "wave$nextNum"
    } else {
        $Wave = "wave1"
    }
}

$WaveOutput = Join-Path $OutputBase $Wave
$WaveDone = Join-Path $DoneBase $Wave
@($WaveOutput, $WaveDone) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}

# â”€â”€ Track our PID â”€â”€
$PID | Set-Content -Path $PidFile

# â”€â”€ State tracking â”€â”€
$script:Tracked = @{}        # pid -> @{Name; Log; StartTime}
$script:LaunchedCount = 0
$script:CompletedCount = 0
$script:FailedCount = 0

# Load existing tracked PIDs
if (Test-Path $TrackedFile) {
    Get-Content $TrackedFile -ErrorAction SilentlyContinue |
        Where-Object { $_ -notmatch '^\s*#' -and $_ -match '\S' } |
        ForEach-Object {
            $parts = $_ -split '\s+', 2
            if ($parts.Count -ge 2) {
                $tPid = [int]$parts[0]
                $tName = $parts[1]
                try {
                    Get-Process -Id $tPid -ErrorAction Stop | Out-Null
                    $script:Tracked[$tPid] = @{ Name = $tName; Log = ""; StartTime = [datetime]::Now }
                } catch {
                    # Process no longer running, skip
                }
            }
        }
}

# â”€â”€ Launch Function â”€â”€
function Start-Experiment {
    param([string]$ConfigBasename)

    $configPath = Join-Path $QueueDir $ConfigBasename
    $expName = Get-ExperimentName $ConfigBasename
    $fullName = "${Wave}_${expName}"
    $expOutput = Join-Path $WaveOutput $expName
    $expLog = Join-Path $LogDir "${fullName}.log"

    if (-not (Test-Path $configPath)) {
        Write-Log "WARNING" "Config $ConfigBasename no longer in queue - skipping"
        return $false
    }

    # Create output directory
    if (-not (Test-Path $expOutput)) {
        New-Item -ItemType Directory -Path $expOutput -Force | Out-Null
    }

    # Validate YAML
    $validateResult = & python -c "import yaml; yaml.safe_load(open(r'$configPath'))" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log "ERROR" "YAML parse error in $ConfigBasename - moving to done with ERROR tag"
        $errorName = ($ConfigBasename -replace '\.yaml$', '') + "_ERROR_$(Get-Date -Format 'yyyyMMdd_HHmmss').yaml"
        Move-Item $configPath (Join-Path $WaveDone $errorName) -Force
        $script:FailedCount++
        return $false
    }

    # Move config to done BEFORE launching (prevents duplicate launches)
    $donePath = Join-Path $WaveDone $ConfigBasename
    Move-Item $configPath $donePath -Force

    # Launch as background process (set UTF-8 to avoid emoji encode errors on Windows)
    $env:GA_OUTPUT_DIR = $expOutput
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    # Ensure TA-Lib DLL directory is on PATH for user site-packages installs
    $talibDir = "C:\Users\peri\AppData\Roaming\Python\Python312\site-packages\talib"
    if ((Test-Path $talibDir) -and ($env:PATH -notlike "*talib*")) {
        $env:PATH = "$talibDir;$env:PATH"
    }
    $proc = Start-Process -FilePath "python" `
        -ArgumentList "genetic_algorithm/run_ga.py --config `"$donePath`" --no-monitor --yes" `
        -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $expLog `
        -RedirectStandardError (Join-Path $LogDir "${fullName}_stderr.log") `
        -PassThru `
        -WindowStyle Hidden

    $procPid = $proc.Id
    $script:Tracked[$procPid] = @{
        Name = $fullName
        Log = $expLog
        StartTime = [datetime]::Now
    }

    # Record in tracked file
    Add-Content -Path $TrackedFile -Value "$procPid $fullName"

    $script:LaunchedCount++
    Save-State -WaveName $Wave -MaxConcurrent $Max -IsPersistent $Persistent.IsPresent `
               -LaunchedCount $script:LaunchedCount -CompletedCount $script:CompletedCount -FailedCount $script:FailedCount

    Write-Log "INFO" "LAUNCHED: $fullName -> PID $procPid"
    Write-Log "INFO" "  Config: $donePath"
    Write-Log "INFO" "  Output: $expOutput"
    Write-Log "INFO" "  Log:    $expLog"
    return $true
}

# â”€â”€ Check Completed Function â”€â”€
function Test-Completed {
    $anyCompleted = $false
    $completedPids = @()

    foreach ($procId in @($script:Tracked.Keys)) {
        $info = $script:Tracked[$procId]
        $isRunning = $false
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            if (-not $proc.HasExited) {
                $isRunning = $true
            }
        } catch {
            # Process not found
        }

        if (-not $isRunning) {
            $name = $info.Name
            $duration = "?"
            if ($info.StartTime) {
                $elapsed = (Get-Date) - $info.StartTime
                $duration = "{0}h{1}m" -f [int]$elapsed.TotalHours, $elapsed.Minutes
            }

            # Try to get exit code
            $exitCode = -1
            try {
                $proc = Get-Process -Id $procId -ErrorAction Stop
                $exitCode = $proc.ExitCode
            } catch {}

            # Extract summary from log
            $bestFitness = "?"
            if ($info.Log -and (Test-Path $info.Log)) {
                $lastLines = Get-Content $info.Log -Tail 200 -ErrorAction SilentlyContinue
                $fitnessLine = $lastLines | Where-Object { $_ -match 'Best:\s+\d' } | Select-Object -Last 1
                if ($fitnessLine) {
                    $null = $fitnessLine -match 'Best:\s+(\d+\.?\d*)'
                    if ($Matches) { $bestFitness = $Matches[1] }
                }
            }

            if ($exitCode -eq 0 -or $exitCode -eq -1) {
                Write-Log "INFO" "COMPLETED: $name ($duration, fitness=$bestFitness)"
                $script:CompletedCount++
            } else {
                Write-Log "WARNING" "FAILED: $name (exit $exitCode, $duration)"
                $script:FailedCount++
            }

            $completedPids += $procId
            $anyCompleted = $true
        }
    }

    foreach ($procId in $completedPids) {
        $script:Tracked.Remove($procId)
    }

    if ($anyCompleted) {
        # Rebuild tracked file
        $lines = @("# Auto-queue tracked PIDs (updated $(Get-Date))")
        foreach ($procId in $script:Tracked.Keys) {
            $lines += "$procId $($script:Tracked[$procId].Name)"
        }
        $lines | Set-Content -Path $TrackedFile -ErrorAction SilentlyContinue
        Save-State -WaveName $Wave -MaxConcurrent $Max -IsPersistent $Persistent.IsPresent `
                   -LaunchedCount $script:LaunchedCount -CompletedCount $script:CompletedCount -FailedCount $script:FailedCount
    }
}

# â”€â”€ Graceful Shutdown Handler â”€â”€
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Write-Log "INFO" "Daemon shutting down - running experiments will continue"
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# â”€â”€ Banner â”€â”€
Write-Host ""
Write-Host "  +================================================================+" -ForegroundColor Cyan
Write-Host "  |   GA Auto-Queue Daemon (PowerShell)                          |" -ForegroundColor Cyan
Write-Host "  +================================================================+" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Wave:            $Wave" -ForegroundColor White
Write-Host "  Max concurrent:  $Max" -ForegroundColor White
Write-Host "  Poll interval:   ${Poll}s" -ForegroundColor White
Write-Host "  Persistent:      $($Persistent.IsPresent)" -ForegroundColor White
Write-Host "  Queue dir:       $QueueDir" -ForegroundColor White
Write-Host "  Done dir:        $WaveDone" -ForegroundColor White
Write-Host "  Output dir:      $WaveOutput" -ForegroundColor White
Write-Host "  Log file:        $LogFile" -ForegroundColor White
Write-Host "  Daemon PID:      $PID" -ForegroundColor White
Write-Host ""

$configs = Get-QueuedConfigs
$queuedCount = if ($configs) { @($configs).Count } else { 0 }
Write-Host "  $queuedCount experiments queued" -ForegroundColor White
if ($configs) {
    foreach ($c in $configs) {
        Write-Host "    -> $($c -replace '\.yaml$', '')" -ForegroundColor DarkGray
    }
}
Write-Host ""

Write-Log "INFO" "Auto-queue daemon started (wave=$Wave, max=$Max, persistent=$($Persistent.IsPresent))"
Save-State -WaveName $Wave -MaxConcurrent $Max -IsPersistent $Persistent.IsPresent `
           -LaunchedCount 0 -CompletedCount 0 -FailedCount 0

# â”€â”€ Idle tracking â”€â”€
$idleSince = $null
$idleNotified = $false

# â”€â”€ Main Loop â”€â”€
try {
    while ($true) {
        Test-Completed

        $currentRunning = $script:Tracked.Count
        $configs = Get-QueuedConfigs
        $queuedCount = if ($configs) { @($configs).Count } else { 0 }

        # Launch new experiments if we have capacity
        $slotsAvailable = $Max - $currentRunning

        if ($slotsAvailable -gt 0 -and $queuedCount -gt 0) {
            $idleSince = $null
            $idleNotified = $false
            $toLaunch = [Math]::Min($slotsAvailable, $queuedCount)
            for ($i = 0; $i -lt $toLaunch; $i++) {
                $nextConfig = (Get-QueuedConfigs | Select-Object -First 1)
                if ($nextConfig) {
                    Start-Experiment $nextConfig | Out-Null
                    Start-Sleep -Seconds 2
                }
            }
        }

        # Check if we're idle
        if ($queuedCount -eq 0 -and $script:Tracked.Count -eq 0) {
            if ($Persistent) {
                if (-not $idleSince) {
                    $idleSince = Get-Date
                }
                if (-not $idleNotified) {
                    Write-Log "INFO" "Queue empty - watching for new configs (persistent mode)"
                    Write-Log "INFO" "Drop YAML files into: $QueueDir"
                    $idleNotified = $true
                }
                # Periodic idle reminder every 5 minutes
                if ($idleSince) {
                    $idleMinutes = [int]((Get-Date) - $idleSince).TotalMinutes
                    if ($idleMinutes -gt 0 -and $idleMinutes % 5 -eq 0) {
                        $lc = $script:LaunchedCount; $cc = $script:CompletedCount; $fc = $script:FailedCount
                        Write-Log "INFO" "Still idle (${idleMinutes}m) - stats: launched=$lc completed=$cc failed=$fc"
                    }
                }
            } else {
                # Non-persistent: double-check then exit
                Start-Sleep -Seconds 5
                $configs = Get-QueuedConfigs
                $queuedCount = if ($configs) { @($configs).Count } else { 0 }
                if ($queuedCount -eq 0 -and $script:Tracked.Count -eq 0) {
                    Write-Log "INFO" "All experiments complete - daemon exiting"
                    $lc = $script:LaunchedCount; $cc = $script:CompletedCount; $fc = $script:FailedCount
                    Write-Log "INFO" "Stats: launched=$lc completed=$cc failed=$fc"
                    Write-Host ""
                    Write-Host "  +=======================================+" -ForegroundColor Green
                    Write-Host "  |   All experiments complete!           |" -ForegroundColor Green
                    Write-Host "  +=======================================+" -ForegroundColor Green
                    Write-Host ""
                    break
                }
            }
        }

        Start-Sleep -Seconds $Poll
    }
} finally {
    $lc = $script:LaunchedCount; $cc = $script:CompletedCount; $fc = $script:FailedCount
    Write-Log "INFO" "Daemon stopping - stats: launched=$lc completed=$cc failed=$fc"
    Save-State -WaveName $Wave -MaxConcurrent $Max -IsPersistent $Persistent.IsPresent `
               -LaunchedCount $script:LaunchedCount -CompletedCount $script:CompletedCount -FailedCount $script:FailedCount
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}
