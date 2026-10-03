# Autonomous Self-Improving AI Coding Agent

An industrial-grade, two-layer autonomous AI coding agent designed with **reliability and recoverability prioritized above autonomy**. The system continuously inspects, enhances, tests, and benchmarks both the project codebase and its own agent software without human intervention, while being strictly governed by a protected supervisor kernel that prevents permanent corruption, infinite loops, runaway processes, or tampering.

---

## Key Highlights

- **Two-Layer Architecture**:
  - **Layer A (Protected Supervisor)**: Unmodifiable guardian. Monitors heartbeats, CPU/RAM, processes, executes health checks, enforces automatic Git rollbacks, and provides a hardware-like kill switch.
  - **Layer B (Autonomous Agent)**: Self-improving coding engine. Inspects architecture, formulates objectives, writes code/tests, benchmarks performance, and can propose modifications to its own planning, memory, and tools.
- **OpenRouter Multi-Key Rotation**:
  - Supports 3–4 API keys (plus a primary default).
  - Automatically detects HTTP 429 (rate-limit), 402 (quota depleted), 401 (invalid), and network timeouts.
  - Transparently switches to the next available key with exponential backoff and cooling periods.
  - Built-in simulation mode (`MOCK_LLM=true`) for local offline development and test execution.
- **Git Safety & Test-Before-Acceptance Rule**:
  - `CHANGE -> TEST -> EVALUATE -> ACCEPT OR ROLLBACK`.
  - Creates a Git checkpoint commit before every modification.
  - Automatically runs pytest and benchmark suites.
  - If tests fail, diagnoses the error and attempts up to 2 automated fixes. If still failing, automatically executes `git reset --hard` back to the last verified commit recorded in `state/known_good_version.json`.
- **Protected Filesystem & Cryptographic Integrity**:
  - The supervisor directory (`/system/supervisor/`) is cryptographically fingerprinted with SHA-256 baseline hashes.
  - Agent tools strictly reject any read, write, delete, or shell command targeting `/system/`.
  - Any unauthorized tampering triggers an immediate alert and rollback.
- **Graceful Self-Restart Protocol**:
  - When the agent modifies its own code, it signals `restart_required = true`.
  - The supervisor verifies the syntax (`py_compile`), gracefully terminates the process, and restarts the agent with persisted state.
- **Persistent State Survival**:
  - Survives reboots, crashes, and restarts via structured JSON state in `state/`.
- **Autonomous Multi-Criteria Objective Selection**:
  - Scores candidates via: `(Impact * 0.25) + (Feasibility * 0.25) + (Safety * 0.20) + (Testability * 0.15) + (MissionAlignment * 0.15) - RedundancyPenalty`.

---

## Directory Structure

```
selfimprover/
├── system/                    # Layer A: Protected Supervisor (Untouchable)
│   ├── supervisor/
│   │   ├── supervisor.py      # Master supervisor process & lifecycle manager
│   │   ├── watchdog.py        # Process, heartbeat, CPU/RAM & runaway monitor
│   │   ├── health_check.py    # 10-point diagnostic health checker
│   │   ├── protection.py      # SHA-256 hash guardian & path sandboxing
│   │   ├── config.yaml        # Supervisor parameters & limits
│   │   └── baseline_hashes.json
│   └── kill_switch.flag       # Tripwire to immediately halt the agent
│
├── agent/                     # Layer B: Autonomous Agent (Self-Modifiable)
│   ├── main.py                # Autonomous continuous loop & heartbeat emitter
│   ├── llm_client.py          # OpenRouter multi-key rotator & failover engine
│   ├── objective_manager.py   # Autonomous objective discovery & scoring
│   ├── planner.py             # Checkpoints, execution, fix retry & rollback
│   ├── evaluator.py           # Test-before-acceptance & benchmark verifier
│   ├── memory.py              # Persistent lessons, experiments & metrics
│   ├── tools.py               # Sandboxed file, test, benchmark & Git tools
│   └── agent_config.yaml      # Agent model parameters & mission
│
├── project/                   # Target Codebase Under Active Improvement
│   ├── src/
│   │   └── ai_pipeline.py     # AI text processing, TF vectorizer & cosine search
│   ├── tests/
│   │   └── test_ai_pipeline.py# Pytest suite
│   ├── benchmarks/
│   │   └── benchmark_pipeline.py # Search latency benchmark
│   └── README.md
│
├── state/                     # Persistent State (Survives Restarts)
│   ├── known_good_version.json # Latest verified Git commit
│   ├── current_objective.json  # In-flight objective (for crash resumption)
│   ├── objectives.json         # Completed & failed objectives history
│   ├── experiments.json        # Detailed experiment logs & diff metrics
│   ├── lessons.json            # Architectural lessons learned
│   ├── metrics.json            # Continuous benchmark & coverage tracking
│   └── heartbeat.json          # Agent heartbeat telemetry for Watchdog
│
├── logs/                      # Multi-Channel Audit Logs
│   ├── supervisor.log         # Supervisor actions, restarts & rollbacks
│   ├── agent.log              # Agent reasoning & cycle progress
│   ├── tool_calls.log         # Every file I/O, test, and shell execution
│   ├── health.log             # Periodic 10-point diagnostic grades
│   └── errors.log             # Exception traces & failure reports
│
├── tests/                     # 27 Unit & End-to-End Integration Tests
├── run.py                     # Master CLI controller & runner
├── .env                       # API key rotation configuration
└── ARCHITECTURE.md            # Detailed architectural specification
```

---

## Quick Start & Setup

### 1. Configure OpenRouter API Keys

Copy `.env.example` to `.env` and insert your OpenRouter API keys:

```bash
# Primary / Default Key
OPENROUTER_API_KEY=sk-or-v1-primary-key-here

# Secondary / Fallback Keys (Rotates automatically if rate-limited or depleted)
OPENROUTER_API_KEY_1=sk-or-v1-fallback-key-1
OPENROUTER_API_KEY_2=sk-or-v1-fallback-key-2
OPENROUTER_API_KEY_3=sk-or-v1-fallback-key-3
OPENROUTER_API_KEY_4=sk-or-v1-fallback-key-4

# Reasoning Model
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct

# Set to true for offline simulation (zero API credit consumption)
MOCK_LLM=false
```

### 2. Inspect System Status

Check overall health, supervisor integrity, last known-good commit, and metrics:

```powershell
python run.py --status
```

Output:
```
--- [System Diagnostic Status] ---
Overall Health: HEALTHY
  [OK]         supervisor_integrity     
  [OK]         agent_liveness           
  [OK]         required_files           
  [OK]         required_packages        
  [OK]         state_accessibility      
  [OK]         memory_system            
  [OK]         test_execution           
  [OK]         api_connectivity         
  [OK]         resource_usage           
  [OK]         child_processes          

Last Known-Good Commit: b3c638b (Initial baseline commit)
Total Completed Experiments: 2
Persistent Lessons Learned: 2
Emergency Kill Switch: INACTIVE (System Ready)
```

### 3. Run the Full Test Suite

Run the 27 unit and end-to-end integration tests:

```powershell
python -m pytest tests/ -v
```

### 4. Start the Autonomous System

Start the Protected Supervisor which manages, monitors, and auto-recovers the Autonomous Agent:

```powershell
# Run continuously under supervisor control
python run.py

# Or run for a specific number of cycles (e.g. 5 cycles)
python run.py --cycles 5
```

---

## Safety & Control Commands

| Command | Action |
| :--- | :--- |
| `python run.py` | Starts the Protected Supervisor and manages the Agent |
| `python run.py --status` | Prints 10-point health check, diagnostics, and recent experiments |
| `python run.py --cycles N` | Runs the agent for N cycles and cleanly stops |
| `python run.py --kill` | **Emergency Kill Switch**: Instantly stops agent subprocesses |
| `python run.py --resume` | Deactivates kill switch and resumes operation |
| `python run.py --rollback` | Manually restores workspace to last known-good commit |
| `python run.py --agent-only` | Runs agent directly in foreground (for development) |
| `python run.py --unsecured` | **Unsecured / Unbound Mode**: Disables supervisor rollback safeguards and layer protections |
| `python run.py --unsecured --agent-only` | Runs agent directly without guardrails or auto-rollback on test failures |

### Unsecured / Unbound Mode (Experimental Sandbox)

For testing in an isolated sandbox or disposable container where you want to see how far the agent evolves without restrictions:

```powershell
python run.py --unsecured
# Or standalone without supervisor:
python run.py --unsecured --agent-only
```

In this mode:
- **Auto-rollback is lifted**: When code modifications break unit tests or benchmarks, the agent does *not* revert; changes are committed anyway, allowing speculative and non-monotonic improvement trajectories.
- **Layer A immutability barrier is lifted**: The agent can inspect, refactor, and self-modify its own tools, supervisor kernel, and architecture.
- **Circuit breaker rollbacks are bypassed**: The supervisor will not force a reset upon repeated failures.
- **Kill switch remains responsive**: You can still halt execution at any moment using `python run.py --kill`.

---

## How It Works

### 1. Autonomous Objective Discovery
The user provides a broad mission (e.g., *"Continuously improve this AI system"*). The agent inspects code coverage, benchmark latency, and past failures in `lessons.json`, formulating prioritized objectives scored by:
- **Impact** (25%)
- **Feasibility** (25%)
- **Safety Risk** (20%)
- **Testability** (15%)
- **Mission Alignment** (15%)
- *Minus Redundancy Penalty if a similar approach recently failed*.

### 2. Checkpoint & Test-Before-Acceptance
Before any modification:
1. Agent creates a Git checkpoint commit (`agent-before-objective-<id>`).
2. Applies modifications to code and tests.
3. Evaluates tests and benchmarks.
4. If tests fail, analyzes errors and attempts up to 2 automated fixes.
5. If still failing: **Automatic rollback** to checkpoint commit.
6. If tests pass: Commits changes and updates `state/known_good_version.json`.

### 3. Multi-Key Failover Engine
If OpenRouter returns:
- `HTTP 429` (Rate Limit): Puts key in temporary cooldown and rotates to next key.
- `HTTP 402` (Payment Required / Credits Depleted): Puts key in 24h cooldown and rotates to next key.
- `HTTP 401` (Unauthorized): Invalidates key and switches to next key.
- Connection Timeout: Switches keys with exponential backoff.

### 4. 10-Point Health Checks
Supervisor periodically verifies:
1. Supervisor integrity (SHA-256 baseline verification).
2. Agent process alive and responsive.
3. Required files exist.
4. Python dependencies importable.
5. Persistent JSON state accessible and non-corrupted.
6. Memory system read/write operational.
7. Pytest runner executable.
8. API connectivity status.
9. RAM & CPU usage within thresholds (< 1024 MB).
10. No runaway child processes (< 15 processes).
