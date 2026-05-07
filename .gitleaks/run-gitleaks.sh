#!/usr/bin/env bash
set -euo pipefail

# Pre-commit hook wrapper for gitleaks secret detection
# This script runs gitleaks in a Docker container using a SHA-pinned image

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Start Docker Desktop and try again"
    exit 1
fi

# Get the script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source the SHA-pinned image version
# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

echo "Running gitleaks secret detection (staged files only)..."
echo "Using image: $GITLEAKS_IMAGE"

# Run gitleaks in Docker container
# - Uses SHA-pinned image for supply chain security
# - Mounts repo as /repo inside container
# - Scans only staged changes (fast, pre-commit appropriate)
# - Exits with gitleaks' exit code (non-zero blocks commit)
docker run --rm -v "$REPO_ROOT:/repo" "$GITLEAKS_IMAGE" \
    protect --staged --source="/repo" --verbose --redact

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ Secret(s) detected in staged files!"
    echo ""
    echo "Remediation steps:"
    echo "1. Remove the secret from the file"
    echo "2. Stage the corrected file: git add <file>"
    echo "3. Try committing again"
    echo ""
    echo "If this is a false positive, add it to .gitleaks.toml allowlist"
    echo ""
    echo "⚠️  Never bypass with --no-verify for actual secrets!"
    echo "   CI will catch bypassed secrets and fail the build."
fi

exit $exit_code
