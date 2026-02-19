#!/bin/bash
# =============================================================================
# WebArena Docker Setup Script
# Automates: download → load → start → configure → verify → auth-cookies
#
# Usage:
#   bash scripts/setup_webarena_docker.sh [--host <hostname>] [--skip-download]
#
# Options:
#   --host <hostname>   Server hostname/IP (default: localhost)
#   --skip-download     Skip image downloads (images already loaded in Docker)
#   --only-start        Only start + configure already-loaded containers
#   --verify            Only run service health checks
#
# Disk space required: ~200 GB for image tars + ~200 GB extracted = ~400 GB
# Time estimate:       Download ~2-8h depending on bandwidth, setup ~30min
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
HOST="localhost"
SKIP_DOWNLOAD=false
ONLY_START=false
ONLY_VERIFY=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEBARENA_DIR="$PROJECT_DIR/webarena"
DOWNLOAD_DIR="/tmp/webarena-images"   # Change to a path with enough space

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)          HOST="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    --only-start)    ONLY_START=true; shift ;;
    --verify)        ONLY_VERIFY=true; shift ;;
    *) error "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Image catalogue ──────────────────────────────────────────────────────────
declare -A IMAGES=(
  ["shopping"]="http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"
  ["shopping_admin"]="http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar"
  ["forum"]="http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"
  ["gitlab"]="http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar"
)
declare -A IMAGE_NAMES=(
  ["shopping"]="shopping_final_0712"
  ["shopping_admin"]="shopping_admin_final_0719"
  ["forum"]="postmill-populated-exposed-withimg"
  ["gitlab"]="gitlab-populated-final-port8023"
)

# ── Step 0: Verify prerequisites ─────────────────────────────────────────────
check_prereqs() {
  info "Checking prerequisites..."
  if ! docker info &>/dev/null; then
    error "Docker daemon is not running. Start it with: sudo systemctl start docker"
    exit 1
  fi
  success "Docker is running"
  if ! command -v curl &>/dev/null; then
    error "curl is required. Install with: apt install curl"
    exit 1
  fi

  # On ARM/aarch64 machines the WebArena images are amd64-only.
  # Install QEMU binfmt handlers if not already present.
  if [[ "$(uname -m)" != "x86_64" ]]; then
    if ! ls /proc/sys/fs/binfmt_misc/ 2>/dev/null | grep -q qemu-x86_64; then
      info "ARM machine detected — installing QEMU x86_64 binfmt handlers..."
      docker run --privileged --rm tonistiigi/binfmt --install all
      success "QEMU x86_64 emulation enabled"
    else
      success "QEMU x86_64 emulation already enabled"
    fi
  fi
}

# ── Step 1: Download image tarballs ──────────────────────────────────────────
download_images() {
  info "Creating download directory: $DOWNLOAD_DIR"
  mkdir -p "$DOWNLOAD_DIR"

  echo ""
  warn "==================================================================="
  warn "  DOWNLOAD SIZE: ~200 GB total"
  warn "  Individual sizes:"
  warn "    shopping:       ~63 GB"
  warn "    shopping_admin:  ~9 GB"
  warn "    forum:          ~50 GB"
  warn "    gitlab:         ~73 GB"
  warn "  Make sure $DOWNLOAD_DIR has enough free space."
  warn "==================================================================="
  echo ""
  read -rp "Continue with download to $DOWNLOAD_DIR? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }

  for service in shopping shopping_admin forum gitlab; do
    url="${IMAGES[$service]}"
    tarfile="$DOWNLOAD_DIR/${IMAGE_NAMES[$service]}.tar"
    if [[ -f "$tarfile" ]]; then
      success "Already downloaded: $tarfile — skipping"
      continue
    fi
    info "Downloading $service from $url ..."
    curl -L --progress-bar --retry 3 --retry-delay 5 -o "$tarfile" "$url"
    success "Downloaded: $tarfile"
  done
}

# ── Step 2: Load images into Docker ──────────────────────────────────────────
load_images() {
  info "Loading Docker images (this stage can take 20-40 min)..."
  for service in shopping shopping_admin forum gitlab; do
    img="${IMAGE_NAMES[$service]}"
    tarfile="$DOWNLOAD_DIR/${img}.tar"

    # Check if image is already loaded
    if docker image inspect "$img" &>/dev/null; then
      success "Image already loaded: $img — skipping"
      continue
    fi

    if [[ ! -f "$tarfile" ]]; then
      error "Tar not found: $tarfile  (re-run without --skip-download)"
      exit 1
    fi

    info "Loading $img ..."
    docker load -i "$tarfile"
    success "Loaded: $img"
  done
}

# ── Step 3: Start containers ─────────────────────────────────────────────────
start_containers() {
  info "Starting containers..."

  # Stop & remove any existing containers first
  for name in shopping shopping_admin forum gitlab; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
      info "Removing existing container: $name"
      docker stop "$name" 2>/dev/null || true
      docker rm   "$name" 2>/dev/null || true
    fi
  done

  # WebArena images are amd64-only; --platform enables QEMU emulation on ARM.
  # On x86_64 machines this flag is a no-op.
  PLATFORM="--platform linux/amd64"

  info "Starting shopping (port 7770)..."
  docker run --name shopping $PLATFORM -p 7770:80 -d shopping_final_0712

  info "Starting shopping_admin (port 7780)..."
  docker run --name shopping_admin $PLATFORM -p 7780:80 -d shopping_admin_final_0719

  info "Starting forum/Reddit (port 9999)..."
  docker run --name forum $PLATFORM -p 9999:80 -d postmill-populated-exposed-withimg

  info "Starting GitLab (port 8023)..."
  docker run --name gitlab $PLATFORM -d -p 8023:8023 gitlab-populated-final-port8023 \
    /opt/gitlab/embedded/bin/runsvdir-start

  info "Starting Wikipedia/Kiwix (port 8888)..."
  # Kiwix ZIM data path — adjust if you downloaded the ZIM file elsewhere
  KIWIX_DATA="${DOWNLOAD_DIR}/kiwix"
  mkdir -p "$KIWIX_DATA"
  if docker ps -a --format '{{.Names}}' | grep -q "^wikipedia$"; then
    docker stop wikipedia 2>/dev/null || true
    docker rm   wikipedia 2>/dev/null || true
  fi
  # If the ZIM file doesn't exist, pull a tiny placeholder or skip
  ZIM_FILE="$KIWIX_DATA/wikipedia_en_all_maxi_2022-05.zim"
  if [[ -f "$ZIM_FILE" ]]; then
    docker run -d --name=wikipedia \
      --volume="${KIWIX_DATA}:/data" \
      -p 8888:80 \
      ghcr.io/kiwix/kiwix-serve:3.3.0 \
      wikipedia_en_all_maxi_2022-05.zim  # Kiwix is multi-arch; no --platform needed
    success "Wikipedia started"
  else
    warn "Wikipedia ZIM not found at $ZIM_FILE — skipping Wikipedia container."
    warn "Download it from: http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim"
    warn "Then re-run this script."
  fi

  success "All containers launched. Waiting 60s for services to initialise..."
  sleep 60
}

# ── Step 4: Configure containers for the target host ─────────────────────────
configure_containers() {
  info "Configuring shopping for host: $HOST ..."
  docker exec shopping \
    /var/www/magento2/bin/magento setup:store-config:set \
    --base-url="http://${HOST}:7770"
  docker exec shopping mysql -u magentouser -pMyPassword magentodb -e \
    "UPDATE core_config_data SET value='http://${HOST}:7770/' WHERE path='web/secure/base_url';"
  docker exec shopping /var/www/magento2/bin/magento cache:flush
  success "Shopping configured"

  info "Configuring shopping_admin for host: $HOST ..."
  docker exec shopping_admin \
    /var/www/magento2/bin/magento setup:store-config:set \
    --base-url="http://${HOST}:7780"
  docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e \
    "UPDATE core_config_data SET value='http://${HOST}:7780/' WHERE path='web/secure/base_url';"
  docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
  # Disable forced password reset
  docker exec shopping_admin php /var/www/magento2/bin/magento \
    config:set admin/security/password_is_forced 0
  docker exec shopping_admin php /var/www/magento2/bin/magento \
    config:set admin/security/password_lifetime 0
  success "Shopping admin configured"

  info "Configuring GitLab for host: $HOST ..."
  docker exec gitlab update-permissions 2>/dev/null || true
  docker exec gitlab sed -i \
    "s|^external_url.*|external_url 'http://${HOST}:8023'|" \
    /etc/gitlab/gitlab.rb
  docker exec gitlab gitlab-ctl reconfigure
  success "GitLab configured (this takes ~5 min the first time)"
}

# ── Step 5: Health check ──────────────────────────────────────────────────────
verify_services() {
  info "Checking service health..."
  declare -A SERVICES=(
    ["Shopping (7770)"]="http://${HOST}:7770"
    ["Shopping Admin (7780)"]="http://${HOST}:7780"
    ["Forum/Reddit (9999)"]="http://${HOST}:9999"
    ["GitLab (8023)"]="http://${HOST}:8023"
    ["Wikipedia (8888)"]="http://${HOST}:8888"
    ["Map (3000)"]="http://${HOST}:3000"
  )
  all_ok=true
  for name in "${!SERVICES[@]}"; do
    url="${SERVICES[$name]}"
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
    if [[ "$code" == "200" || "$code" == "302" || "$code" == "301" ]]; then
      success "$name → HTTP $code ✓"
    else
      warn    "$name → HTTP $code ✗  ($url)"
      all_ok=false
    fi
  done
  $all_ok && success "All services healthy!" || warn "Some services may still be starting — retry in a minute."
}

# ── Step 6: Prepare auth cookies ─────────────────────────────────────────────
prepare_auth() {
  info "Preparing authentication cookies..."
  cd "$WEBARENA_DIR"
  if [[ -f prepare.sh ]]; then
    SHOPPING="http://${HOST}:7770" \
    SHOPPING_ADMIN="http://${HOST}:7780/admin" \
    REDDIT="http://${HOST}:9999" \
    GITLAB="http://${HOST}:8023" \
    MAP="http://${HOST}:3000" \
    bash prepare.sh
    success "Auth cookies generated in webarena/.auth/"
  else
    warn "prepare.sh not found — skipping auth cookie generation."
  fi
  cd "$PROJECT_DIR"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  echo ""
  echo -e "${CYAN}======================================================${NC}"
  echo -e "${CYAN}  WebArena Docker Setup — host: $HOST${NC}"
  echo -e "${CYAN}======================================================${NC}"
  echo ""

  check_prereqs

  if $ONLY_VERIFY; then
    verify_services
    exit 0
  fi

  if $ONLY_START; then
    load_images
    start_containers
    configure_containers
    verify_services
    prepare_auth
    exit 0
  fi

  if ! $SKIP_DOWNLOAD; then
    download_images
  fi

  load_images
  start_containers
  configure_containers
  verify_services
  prepare_auth

  echo ""
  success "=== WebArena setup complete! ==="
  echo ""
  info  "Quick test:"
  echo "  source ~/.venv/bin/activate"
  echo "  python -m scripts.collect_trajectories \\"
  echo "    --config configs/webarena/clean.yaml \\"
  echo "    collection.num_episodes=5"
}

main
