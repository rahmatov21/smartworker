#!/usr/bin/env bash
# ==============================================================================
# Autonomous Self-Improving AI Coding Agent (AGY)
# Server Startup Script - Unsecured Continuous Background Mode
# ==============================================================================

set -e

# Change directory to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Self-Improver in UNSECURED Mode on Server"
echo " Working Directory: $SCRIPT_DIR"
echo "======================================================================"

# 1. Check Python & Git
command -v python3 >/dev/null 2>&1 || { echo "[!] Python3 is required but not installed. Aborting."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "[!] Git is required but not installed. Aborting."; exit 1; }

# 2. Configure Git user if not already set (required for autonomous commits)
if [ -z "$(git config --get user.name)" ]; then
    echo "[*] Configuring Git bot identity..."
    git config user.name "AutonomousAgent"
    git config user.email "agent@selfimprover.internal"
fi

# 3. Virtual Environment setup
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
else
    echo "[*] Virtual environment found."
fi

# 4. Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[!] .env not found. Creating from .env.example..."
        cp .env.example .env
        echo "[!] Please configure your OpenRouter API keys in .env before running!"
    fi
fi

# 5. Create logs and state directories
mkdir -p logs state system

# 6. Ensure kill switch is inactive
rm -f system/kill_switch.flag

# 7. Mode Selection: Check if tmux is available
if command -v tmux >/dev/null 2>&1; then
    SESSION_NAME="selfimprover"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "[!] A tmux session '$SESSION_NAME' is already running!"
        echo "    To attach: tmux attach -t $SESSION_NAME"
        exit 0
    fi

    echo "[+] Launching in persistent tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME" "./venv/bin/python -u run.py --unsecured"
    echo "======================================================================"
    echo " Agent is now running continuously in the background!"
    echo " - Attach to live terminal: tmux attach -t $SESSION_NAME"
    echo " - Detach from terminal:    Press Ctrl+B then D"
    echo " - Check system status:     ./venv/bin/python run.py --status"
    echo " - Check healer/stalls:     ./venv/bin/python run.py --heal-now"
    echo " - Emergency kill switch:   ./venv/bin/python run.py --kill"
    echo " - Tail audit logs:         tail -f logs/agent.log"
    echo "======================================================================"
else
    # Fallback to nohup background process
    echo "[*] tmux not found. Launching via nohup in background..."
    nohup ./venv/bin/python -u run.py --unsecured > logs/run.log 2>&1 &
    PID=$!
    echo "[+] Agent started under PID $PID. Logs: logs/run.log"
    echo "    To monitor: tail -f logs/agent.log"
fi
