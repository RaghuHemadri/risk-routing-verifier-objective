# Setup Instructions

This guide walks through the full environment setup required to run the
R2V-Agent pipeline (trajectory collection → perturbation → training → evaluation).

> **Tested on:** Ubuntu/Debian, Python 3.12, Docker 29+

---

## 1. Python Environment

Python **3.12** is required (the project requires ≥ 3.10; system Python on many
machines is 3.9 which will not work).

```bash
# Create venv using python3.12 explicitly
python3.12 -m venv ~/.venv
source ~/.venv/bin/activate

# Upgrade pip and setuptools first (required — older setuptools break the install)
pip install --upgrade pip setuptools

# Install the package with all dependencies
pip install -e ".[dev]"
```

> **Note:** The `pyproject.toml` uses `build-backend = "setuptools.build_meta"`.
> If you cloned before this was fixed (it originally had `setuptools.backends._legacy`),
> ensure the first three lines of `pyproject.toml` read:
> ```toml
> [build-system]
> requires = ["setuptools>=45", "wheel"]
> build-backend = "setuptools.build_meta"
> ```

---

## 2. API Keys

```bash
cp .env.example .env
```

Edit `.env` and set at least one LLM key:

| Variable            | Provider  | Sign-up URL                              |
|---------------------|-----------|------------------------------------------|
| `OPENAI_API_KEY`    | OpenAI    | https://platform.openai.com/api-keys     |
| `ANTHROPIC_API_KEY` | Anthropic | https://console.anthropic.com/           |
| `GOOGLE_API_KEY`    | Google    | https://aistudio.google.com/apikey       |
| `DEEPSEEK_API_KEY`  | DeepSeek  | https://platform.deepseek.com/api_keys   |

The teacher model is configured in `configs/base.yaml`:

```yaml
teacher:
  provider: openai          # openai | anthropic | google | deepseek
  model_name: gpt-4o
```

---

## 3. Smoke Tests (no Docker needed)

Run these immediately after step 2 to confirm the full pipeline works.
Both use `gpt-4o-mini` — total cost under **$0.001**.

### SWE-bench smoke test

Downloads `princeton-nlp/SWE-bench_Lite` (~1.2 MB) and runs 1 task, 3 seeds.
No Docker required.

```bash
source ~/.venv/bin/activate

python -m scripts.collect_trajectories \
    --config configs/smoke_test_swebench.yaml \
    --output data/smoke_test_swebench \
    --num-episodes 1
```

Expected output:
```
[INFO] Teacher: provider=openai, model=gpt-4o-mini
[INFO] Loaded 1 SWE-bench instances
[INFO] Loaded 1 tasks, collecting with seeds=[1, 2, 3]
[INFO] Collection complete — Episodes: 3
```

### WebArena smoke test

Uses the real Playwright browser against a live `shopping_admin` Docker
container (~9 GB). Only this one service is needed — tasks 0, 1, 11, etc.
all target `shopping_admin` only.

**One-time setup (do this before the smoke test):**

```bash
# 1. Install QEMU emulation if on an ARM/aarch64 machine
#    (the WebArena images are amd64; skip on x86_64 machines)
docker run --privileged --rm tonistiigi/binfmt --install all

# 2. Download and load the shopping_admin image (~9 GB)
curl -L --retry 3 -o /tmp/shopping_admin_final_0719.tar \
  http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar
docker load -i /tmp/shopping_admin_final_0719.tar

# 3. Start the container (add --platform linux/amd64 on ARM machines)
docker run --name shopping_admin --platform linux/amd64 -p 7780:80 -d shopping_admin_final_0719
sleep 90   # wait for Magento to boot

# 4. Configure Magento base URL
docker exec shopping_admin \
  /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \
  "UPDATE core_config_data SET value='http://localhost:7780/' WHERE path='web/secure/base_url';"
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0

# 5. Generate auth cookie
cd webarena
python browser_env/auto_login.py --site_list shopping_admin
cd ..
ln -sfn webarena/.auth .auth   # make cookie accessible from project root
```

**Run the smoke test:**

```bash
source ~/.venv/bin/activate

python -m scripts.collect_trajectories \
    --config configs/smoke_test_webarena.yaml \
    --output data/smoke_test_webarena \
    --num-episodes 1
```

Expected output:
```
[INFO] Teacher: provider=openai, model=gpt-4o-mini
[INFO] Found 814 WebArena task configs in ./config_files
[INFO] Loaded 1 WebArena tasks
[INFO] Collecting trajectory: task=0, seed=1
[INFO] Collection complete — Episodes: 3
```

The browser launches headlessly, loads the Magento admin dashboard, and the
LLM takes real steps. Output saved to `data/smoke_test_webarena/trajectories.jsonl`.

---

## 4. WebArena Setup

WebArena requires self-hosted Docker containers (~200 GB of images) and
pre-generated task config files.

### 4a. Clone and install the WebArena repo

```bash
cd <project-root>
git clone https://github.com/web-arena-x/webarena.git

# Install extra runtime deps WebArena needs
source ~/.venv/bin/activate
pip install beartype==0.12.0 gymnasium flask aiolimiter evaluate nltk text-generation

# Install the webarena package itself (provides the browser_env module)
pip install -e webarena/
```

### 4b. Install Playwright browser

```bash
pip install playwright
python -m playwright install chromium
```

### 4c. Generate task config files (815 tasks)

```bash
cd webarena

# Set env vars pointing at localhost (update host if using a remote machine)
export SHOPPING=http://localhost:7770
export SHOPPING_ADMIN=http://localhost:7780/admin
export REDDIT=http://localhost:9999
export GITLAB=http://localhost:8023
export MAP=http://localhost:3000
export WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kev/Wikipedia:Village_Pump/Proposals/Sysop_teleportation"
export HOMEPAGE=http://localhost:4399

python scripts/generate_test_data.py
# → generates config_files/*.json (815 files)

cd ..
# Symlink into project root
ln -sfn webarena/config_files config_files
```

Update `.env`:
```
WEBARENA_CONFIG_DIR=./config_files
```

### 4d. Launch Docker services (~200 GB)

Use the provided automated script which handles download → load → start →
configure → health check → auth cookies:

```bash
# Full setup (prompts before downloading ~200 GB)
bash scripts/setup_webarena_docker.sh

# If services are already downloaded/loaded, just start and configure them:
bash scripts/setup_webarena_docker.sh --only-start

# Health check only:
bash scripts/setup_webarena_docker.sh --verify
```

Image sizes:
| Service         | Image                              | Size  |
|-----------------|------------------------------------|-------|
| Shopping        | `shopping_final_0712`              | ~63 GB |
| Shopping Admin  | `shopping_admin_final_0719`        | ~9 GB  |
| Forum (Reddit)  | `postmill-populated-exposed-withimg` | ~50 GB |
| GitLab          | `gitlab-populated-final-port8023`  | ~73 GB |

After containers are running, `.env` already has the correct localhost URLs.
Auth cookies are written to `webarena/.auth/` by the script.

---

## 5. SWE-bench Setup

SWE-bench needs Docker only for evaluation. Trajectory *collection* works
without Docker. The dataset downloads automatically from HuggingFace.

```bash
# Verify Docker is available (only needed for evaluation, not collection)
docker info
```

No other manual steps required — the dataset is downloaded on first run.

---

## 6. Running the Pipeline

### Collect trajectories

```bash
source ~/.venv/bin/activate

# SWE-bench (no Docker required for collection)
python -m scripts.collect_trajectories \
    --config configs/swebench/clean.yaml \
    --output data/trajectories/swebench \
    --num-episodes 5

# WebArena (Docker services must be running)
python -m scripts.collect_trajectories \
    --config configs/webarena/clean.yaml \
    --output data/trajectories/webarena \
    --num-episodes 5

# Override teacher model on the command line
python -m scripts.collect_trajectories \
    --config configs/swebench/clean.yaml \
    --output data/trajectories/swebench_claude \
    --num-episodes 5 \
    --overrides teacher.provider=anthropic teacher.model_name=claude-3-5-sonnet-20241022
```

### Full pipeline via SLURM

```bash
bash scripts/slurm/run_all.sh
```

Or stage by stage:

```bash
sbatch scripts/slurm/01_collect.sh
sbatch scripts/slurm/02_perturb.sh
sbatch scripts/slurm/03_train_policy.sh
sbatch scripts/slurm/04_train_verifier.sh
sbatch scripts/slurm/05_generate_candidates.sh
sbatch scripts/slurm/06_generate_router_features.sh
sbatch scripts/slurm/07_train_router.sh
sbatch scripts/slurm/08_evaluate.sh
sbatch scripts/slurm/09_ablations.sh
```

---

## 7. Directory Structure After Setup

```
project/
├── .env                      # API keys (git-ignored)
├── .env.example              # Template
├── config_files/             # Symlink → webarena/config_files/ (815 JSONs)
├── webarena/                 # Cloned WebArena repo
├── data/
│   ├── smoke_test_swebench/   # Output of SWE-bench smoke test
│   ├── smoke_test_webarena/   # Output of WebArena smoke test
│   ├── trajectories/         # Collected episodes (JSONL)
│   ├── perturbations/        # Perturbed episodes
│   └── checkpoints/          # Trained model weights
├── configs/
│   ├── base.yaml
│   ├── smoke_test_swebench.yaml   # 1 instance, no Docker
│   ├── smoke_test_webarena.yaml   # 1 task, mock mode (no Docker)
│   ├── swebench/
│   └── webarena/
├── r2v/                      # Core library
└── scripts/
    ├── setup_webarena_docker.sh  # Automated WebArena Docker setup
    └── slurm/                    # SLURM job scripts
```

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `BackendUnavailable: Cannot import 'setuptools.backends._legacy'` | Check `pyproject.toml` uses `build-backend = "setuptools.build_meta"` (not `._legacy`) and run `pip install --upgrade setuptools` |
| `Package requires Python >=3.10` | Your venv uses Python 3.9. Create it with `python3.12 -m venv ~/.venv` |
| `exec format error` when starting WebArena container | Machine is ARM/aarch64 but images are amd64. Run: `docker run --privileged --rm tonistiigi/binfmt --install all` then add `--platform linux/amd64` to `docker run` |
| `ModuleNotFoundError: browser_env` | Run `pip install -e webarena/` from the project root |
| `ModuleNotFoundError: beartype` | Run `pip install beartype==0.12.0 gymnasium flask` |
| `playwright: command not found` | Use `python -m playwright install chromium` (not the system `playwright`) |
| `docker.errors.DockerException` | Run `sudo systemctl start docker` |
| `OPENAI_API_KEY not set` | Check `.env` exists and is not the `.env.example` template |
| SWE-bench evaluation hangs | Increase Docker memory limit; check `docker stats` |
| WebArena task configs missing | Re-run step 4c with the env vars exported |
