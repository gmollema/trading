#!/usr/bin/env pwsh
# RSI2 Trading System - Smart Daily Check with ANSI colors

Set-Location C:\Users\gijs\PythonProjects\ibkr

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RSI2 TRADING SYSTEM - Daily Check" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if positions file exists and has open positions
$positionsFile = "rsi2_open_positions.json"
$hasOpenPositions = $false

if (Test-Path $positionsFile) {
    $content = Get-Content $positionsFile -Raw
    $positions = $content | ConvertFrom-Json
    $hasOpenPositions = ($positions -is [array] -and $positions.Count -gt 0) -or ($positions -is [object] -and $positions.psobject.properties.count -gt 0)
}

if ($hasOpenPositions) {
    Write-Host "OPEN POSITIONS DETECTED - Showing position monitor only" -ForegroundColor Yellow
    Write-Host ""
    python -m trading_bot.cli.rsi2_position_monitor
} else {
    Write-Host "NO OPEN POSITIONS - Showing entry signals only" -ForegroundColor Green
    Write-Host ""
    python -m trading_bot.cli.rsi2_daily_signals
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Done! Act on any colored signals above" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close..."
