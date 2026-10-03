# 24/7 Server Deployment Guide (Unsecured Mode)

This guide explains how to deploy and keep the autonomous self-improving agent running 24/7 on a remote Linux server (e.g. DigitalOcean, AWS, Linode, Hetzner, or a private VPS) in **Unsecured Mode** (`--unsecured`), surviving terminal disconnections, SSH timeouts, and server reboots.

---

## Step 1: Copy Code to the Server

From your local machine (PowerShell / Terminal):

```powershell
# Using scp to transfer the workspace to the server:
scp -r "c:\Users\Abdulaziz\Desktop\All codes\selfimprover" root@<YOUR_SERVER_IP>:/root/selfimprover
```

*Or if your code is pushed to a private Git repository:*
```bash
ssh root@<YOUR_SERVER_IP>
git clone <YOUR_GIT_REPO_URL> /root/selfimprover
```

---

## Step 2: SSH into the Server & Install Prerequisites

Connect to the server:
```bash
ssh root@<YOUR_SERVER_IP>
cd /root/selfimprover
```

Install system packages:
```bash
apt update && apt install -y python3 python3-venv python3-pip git tmux
```

Configure Git identity (required for the agent to commit improvements autonomously):
```bash
git config --global user.name "AutonomousAgent"
git config --global user.email "agent@selfimprover.internal"
```

Set up virtual environment:
```bash
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

---

## Step 3: Configure OpenRouter Keys

Edit `.env` and paste your OpenRouter API keys:
```bash
nano .env
```
*(Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`)*

---

## Step 4: Choose How to Keep It Running 24/7

Choose **Method A** (easiest for interactive monitoring) or **Method B** (best for permanent 24/7 background operation).

### Method A: Using `tmux` (Recommended for Interactive Monitoring)

`tmux` allows processes to run in a virtual terminal session that stays alive even when you close your SSH connection.

1. **Start the session using the included helper**:
   ```bash
   chmod +x start_unsecured.sh
   ./start_unsecured.sh
   ```

2. **Or start tmux manually**:
   ```bash
   tmux new -s selfimprover
   ./venv/bin/python -u run.py --unsecured
   ```

3. **Detach from the session** (leaves it running in background):
   - Press `Ctrl + B`, release both keys, then press `D`.
   - You can now safely close your SSH terminal.

4. **Re-attach to the live running agent anytime**:
   ```bash
   tmux attach -t selfimprover
   ```

---

### Method B: Using `systemd` (Recommended for Production & Reboot Survival)

`systemd` manages the agent as a system service: it starts automatically when the server boots and restarts automatically if terminated.

1. **Copy the service file**:
   ```bash
   cp selfimprover.service /etc/systemd/system/
   ```

2. **Reload systemd & start service**:
   ```bash
   systemctl daemon-reload
   systemctl enable selfimprover
   systemctl start selfimprover
   ```

3. **Check service status**:
   ```bash
   systemctl status selfimprover
   ```

4. **View live output**:
   ```bash
   journalctl -u selfimprover -f
   # Or tail file logs:
   tail -f /root/selfimprover/logs/service.log
   ```

5. **Stop or Restart the service**:
   ```bash
   systemctl stop selfimprover
   systemctl restart selfimprover
   ```

---

### Method C: Using `nohup` (Simple Background Process)

```bash
nohup ./venv/bin/python -u run.py --unsecured > logs/run.log 2>&1 &
```
To check if running:
```bash
ps aux | grep run.py
```

---

## Step 5: Remote Control & Monitoring Commands

While the agent runs on the server, you can run these commands from another SSH session:

| Action | Command |
| :--- | :--- |
| **Inspect System Diagnostics** | `./venv/bin/python run.py --status` |
| **Trigger Auto-Healer Scan & Fix** | `./venv/bin/python run.py --heal-now` |
| **Follow Live Agent Thoughts** | `tail -f logs/agent.log` |
| **Follow Supervisor Audit Log** | `tail -f logs/supervisor.log` |
| **Follow AutoHealer Actions** | `tail -f logs/healer.log` |
| **View Recent Completed Experiments** | `cat state/experiments.json` |
| **Emergency Halt (Kill Switch)** | `./venv/bin/python run.py --kill` |
| **Resume After Halt** | `./venv/bin/python run.py --resume` |
