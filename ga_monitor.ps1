<#
.SYNOPSIS
    GA Evolution Monitor for Windows — Live dashboard for running experiments.

.DESCRIPTION
    Displays a live-updating dashboard showing all running GA experiments,
    generation progress, best fitness, runtime, and queue status.

.PARAMETER Wave
    Filter to a specific wave name (e.g. wave26_laptop). Default: show all.

.PARAMETER Interval
    Refresh interval in seconds (default: 5).

.PARAMETER Once
    Print once and exit (no auto-refresh).

.EXAMPLE
    .\ga_monitor.ps1
    .\ga_monitor.ps1 -Wave wave26_laptop
    .\ga_monitor.ps1 -Once
    .\ga_monitor.ps1 -Interval 10
#>

param(
    [string]$Wave = "",
    [int]$Interval = 5,
    [switch]$Once
)

$RepoDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir     = Join-Path $RepoDir "genetic_algorithm\logs"
$QueueDir   = Join-Path $RepoDir "genetic_algorithm\config\queue"
$DoneBase   = Join-Path $RepoDir "genetic_algorithm\config\done"
$StateFile  = Join-Path $LogDir "auto_queue_state.json"
$TrackedFile= Join-Path $LogDir "auto_queue_tracked.txt"

# ── Helpers ──

function Format-Duration {
    param([timespan]$ts)
    if ($ts.TotalHours -ge 1) { return "{0}h{1:00}m" -f [int]$ts.TotalHours, $ts.Minutes }
    if ($ts.TotalMinutes -ge 1) { return "{0}m{1:00}s" -f [int]$ts.TotalMinutes, $ts.Seconds }
    return "{0}s" -f [int]$ts.TotalSeconds
}

function Get-ProgressBar {
    param([int]$Current, [int]$Total, [int]$Width = 20)
    if ($Total -le 0) { return "[" + ("?" * $Width) + "]" }
    $filled = [int]([Math]::Round(($Current / $Total) * $Width))
    $filled = [Math]::Max(0, [Math]::Min($Width, $filled))
    $empty  = $Width - $filled
    return "[" + ("#" * $filled) + ("-" * $empty) + "]"
}

function Parse-LogInfo {
    param([string]$LogPath)

    $info = @{
        CurrentGen  = 0
        TotalGens   = 0
        BestFitness = $null
        BestProfit  = $null
        LastActivity= $null
        Phase       = "starting"
        IslandCount = 0
        Status      = "starting"
        PopPerIsland= 0
    }

    if (-not (Test-Path $LogPath)) { return $info }

    $lines = Get-Content $LogPath -ErrorAction SilentlyContinue
    if (-not $lines -or $lines.Count -eq 0) { return $info }

    foreach ($line in $lines) {
        # Total generations
        if ($line -match 'Generations:\s+(\d+)') {
            $info.TotalGens = [int]$Matches[1]
        }
        # Islands
        if ($line -match 'Islands:\s+(\d+)') {
            $info.IslandCount = [int]$Matches[1]
        }
        # Pop per island
        if ($line -match 'Pop/island:\s+(\d+)') {
            $info.PopPerIsland = [int]$Matches[1]
        }
        # Current generation
        if ($line -match 'GENERATION (\d+)/(\d+)') {
            $info.CurrentGen = [int]$Matches[1]
            $info.TotalGens  = [int]$Matches[2]
            $info.Phase      = "evolving"
        }
        # Best fitness from NEW BEST line
        if ($line -match 'NEW BEST: fitness=(\d+\.\d+)') {
            $f = [double]$Matches[1]
            if ($null -eq $info.BestFitness -or $f -gt $info.BestFitness) {
                $info.BestFitness = $f
                # Profit on same line
                if ($line -match 'profit=([+-]?\d+\.\d+%)') {
                    $info.BestProfit = $Matches[1]
                }
            }
        }
        # Island summary: best=X.XXXX (fallback if no NEW BEST seen yet)
        if ($line -match '\] best=(\d+\.\d+) ') {
            $f = [double]$Matches[1]
            if ($null -eq $info.BestFitness -or $f -gt $info.BestFitness) {
                $info.BestFitness = $f
            }
        }
        # Phase detection
        if ($line -match 'PHASE 1') { $info.Phase = "init" }
        if ($line -match 'PHASE 2') { $info.Phase = "evolving" }
        if ($line -match 'PHASE 3|MERGE|merge round') { $info.Phase = "merging" }
        if ($line -match 'Evolution complete|EVOLUTION COMPLETE|hall.of.fame') { $info.Phase = "complete" }
    }

    # Last activity timestamp from last line
    $lastLine = $lines | Select-Object -Last 1
    if ($lastLine -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})') {
        try { $info.LastActivity = [datetime]$Matches[1] } catch {}
    }

    if ($info.Phase -eq "complete") { $info.Status = "done" }
    elseif ($info.CurrentGen -gt 0) { $info.Status = "running" }
    elseif ($info.Phase -eq "init") { $info.Status = "init" }
    else { $info.Status = "starting" }

    return $info
}

function Get-FitnessColor {
    param([double]$f)
    if ($f -ge 0.70) { return "Green" }
    if ($f -ge 0.55) { return "Cyan" }
    if ($f -ge 0.40) { return "Yellow" }
    return "DarkGray"
}

function Get-PhaseLabel {
    param([string]$phase)
    switch ($phase) {
        "starting"  { return "STARTING" }
        "init"      { return "INIT    " }
        "evolving"  { return "EVOLVING" }
        "merging"   { return "MERGE   " }
        "complete"  { return "DONE    " }
        default     { return "UNKNOWN " }
    }
}

function Show-Dashboard {
    param([switch]$ClearScreen)
    $now = Get-Date

    # ── Load state ──
    $daemonRunning = $false
    $waveName = "unknown"
    $daemonPid = $null
    $launchedCount = 0
    $completedCount = 0
    $failedCount = 0

    $pidFile = Join-Path $LogDir "auto_queue_daemon.pid"
    if (Test-Path $pidFile) {
        $daemonPid = (Get-Content $pidFile -ErrorAction SilentlyContinue) -as [int]
        if ($daemonPid) {
            try { Get-Process -Id $daemonPid -ErrorAction Stop | Out-Null; $daemonRunning = $true } catch {}
        }
    }

    if (Test-Path $StateFile) {
        try {
            $state = Get-Content $StateFile | ConvertFrom-Json
            $waveName    = $state.wave
            $launchedCount  = $state.launched_count
            $completedCount = $state.completed_count
            $failedCount    = $state.failed_count
        } catch {}
    }

    # ── Discover log files (GA output goes to _stderr.log via Python logging) ──
    $pattern = if ($Wave) { "*${Wave}*_stderr.log" } else { "*_stderr.log" }
    $logFiles = Get-ChildItem -Path $LogDir -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '^auto_queue' } |
        Sort-Object Name

    # ── Load tracked PIDs ──
    $trackedPids = @{}
    if (Test-Path $TrackedFile) {
        Get-Content $TrackedFile -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^\d+ \S+' } |
            ForEach-Object {
                $parts = $_ -split '\s+', 2
                if ($parts.Count -ge 2) { $trackedPids[[int]$parts[0]] = $parts[1] }
            }
    }

    # ── Queue ──
    $queued = @(Get-ChildItem -Path $QueueDir -Filter "*.yaml" -ErrorAction SilentlyContinue | Sort-Object Name)
    $queuedCount = $queued.Count

    # ── Header ──
    if ($ClearScreen) { Clear-Host }
    $width = $Host.UI.RawUI.WindowSize.Width
    if ($width -lt 80) { $width = 110 }

    Write-Host ""
    Write-Host ("  +{0}+" -f ("=" * ($width - 4))) -ForegroundColor Cyan
    $title = "  GA Evolution Monitor"
    $ts    = $now.ToString("HH:mm:ss")
    $pad   = $width - 4 - $title.Length - $ts.Length - 2
    Write-Host ("  | {0}{1}{2} |" -f $title, (" " * [Math]::Max(0,$pad)), $ts) -ForegroundColor Cyan
    Write-Host ("  +{0}+" -f ("=" * ($width - 4))) -ForegroundColor Cyan
    Write-Host ""

    # ── Daemon status bar ──
    $dStatus = if ($daemonRunning) { "* RUNNING (PID $daemonPid)" } else { "o STOPPED" }
    $dColor  = if ($daemonRunning) { "Green" } else { "Yellow" }
    Write-Host "  Daemon: " -NoNewline -ForegroundColor DarkGray
    Write-Host $dStatus -NoNewline -ForegroundColor $dColor
    Write-Host "   Wave: $waveName   Launched: $launchedCount   Done: $completedCount   Failed: $failedCount   Queue: $queuedCount" -ForegroundColor DarkGray
    Write-Host ""

    # ── Experiments table ──
    if ($logFiles.Count -eq 0) {
        Write-Host "  No experiment logs found." -ForegroundColor DarkGray
    } else {
        # Header row
        $h = "  {0,-30} {1,-6} {2,-8} {3,-22} {4,-10} {5,-8} {6,-10} {7}" -f `
            "EXPERIMENT", "STATUS", "PHASE", "PROGRESS", "FITNESS", "PROFIT", "RUNTIME", "LAST UPDATE"
        Write-Host $h -ForegroundColor White
        Write-Host ("  " + "-" * ($width - 4)) -ForegroundColor DarkGray

        foreach ($logFile in $logFiles) {
            $expName = $logFile.BaseName -replace '_stderr$', ''

            # Check if process is running (match tracked PIDs to log name)
            $trackedEntry = $trackedPids.GetEnumerator() | Where-Object { $_.Value -eq $expName } | Select-Object -First 1
            $isAlive = $false
            if ($trackedEntry) {
                try { Get-Process -Id $trackedEntry.Key -ErrorAction Stop | Out-Null; $isAlive = $true } catch {}
            }

            $info = Parse-LogInfo -LogPath $logFile.FullName

            # Runtime from log file age vs start
            $startTime = $logFile.CreationTime
            $runtime = Format-Duration ($now - $startTime)

            # Status
            if ($info.Phase -eq "complete") {
                $statusLabel = "DONE  "
                $statusColor = "Green"
            } elseif ($isAlive) {
                $statusLabel = "LIVE  "
                $statusColor = "Cyan"
            } elseif ($logFile.Length -eq 0) {
                $statusLabel = "CRASH "
                $statusColor = "Red"
            } else {
                $statusLabel = "DEAD  "
                $statusColor = "Yellow"
            }

            # Progress bar
            $genText = if ($info.TotalGens -gt 0) {
                "{0,3}/{1,-3}" -f $info.CurrentGen, $info.TotalGens
            } else { "  ?/?" }
            $bar = Get-ProgressBar -Current $info.CurrentGen -Total $info.TotalGens -Width 12

            # Fitness
            if ($null -ne $info.BestFitness) {
                $fitnessText  = "{0:F4}" -f $info.BestFitness
                $fitnessColor = Get-FitnessColor -f $info.BestFitness
            } else {
                $fitnessText  = "  ---"
                $fitnessColor = "DarkGray"
            }

            $profitText = if ($info.BestProfit) { $info.BestProfit } else { "---" }

            # Last activity
            $lastUpdate = if ($info.LastActivity) { $info.LastActivity.ToString("HH:mm:ss") } else { "---" }

            # Short name (strip wave prefix)
            $shortName = $expName -replace "^wave[^_]+_", ""

            # Print row
            Write-Host ("  {0,-30} " -f $shortName) -NoNewline -ForegroundColor White
            Write-Host ("{0,-6} " -f $statusLabel) -NoNewline -ForegroundColor $statusColor
            Write-Host ("{0,-8} " -f (Get-PhaseLabel $info.Phase)) -NoNewline -ForegroundColor DarkGray
            Write-Host ("{0} {1} " -f $bar, $genText) -NoNewline -ForegroundColor DarkGray
            Write-Host ("{0,-10} " -f $fitnessText) -NoNewline -ForegroundColor $fitnessColor
            Write-Host ("{0,-8} " -f $profitText) -NoNewline -ForegroundColor DarkGray
            Write-Host ("{0,-10} " -f $runtime) -NoNewline -ForegroundColor DarkGray
            Write-Host $lastUpdate -ForegroundColor DarkGray
        }
    }

    # ── Queue section ──
    Write-Host ""
    Write-Host ("  " + "-" * ($width - 4)) -ForegroundColor DarkGray
    if ($queuedCount -gt 0) {
        Write-Host "  Queued ($queuedCount):" -ForegroundColor Yellow
        foreach ($q in $queued) {
            Write-Host ("    -> {0}" -f ($q.Name -replace '\.yaml$', '')) -ForegroundColor DarkGray
        }
    } else {
        Write-Host "  Queue empty" -ForegroundColor DarkGray
    }

    # ── Controls ──
    Write-Host ""
    Write-Host ("  " + "-" * ($width - 4)) -ForegroundColor DarkGray
    if ($Once) {
        Write-Host "  Snapshot at $($now.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor DarkGray
    } else {
        Write-Host ("  Refreshing every ${Interval}s  |  Ctrl+C to exit  |  .\ga_monitor.ps1 -Once for snapshot") -ForegroundColor DarkGray
    }
    Write-Host ""
}

# ── Main ──
if ($Once) {
    Show-Dashboard
} else {
    try {
        while ($true) {
            Show-Dashboard -ClearScreen
            Start-Sleep -Seconds $Interval
        }
    } catch [System.Management.Automation.PipelineStoppedException] {
        # Ctrl+C — clean exit
    }
}
