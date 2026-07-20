#!/usr/bin/env bash
# Sparse checkout — download only dashboard/ + reports/release/
# Usage:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/BobbyWeng/cannbot-ascendc-vs-pypto/<BRANCH>/dashboard/download_dashboard.sh)" _ <repo-url>
#
# Then:
#   cd cannbot-dashboard && python3 dashboard/dashboard.py --release
#   open dashboard/index.html

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <repo-url>"
    exit 1
fi

REPO_URL="$1"
DIR="cannbot-dashboard"

echo "==> Cloning with sparse checkout: $REPO_URL"
git clone --filter=blob:none --no-checkout "$REPO_URL" "$DIR"
cd "$DIR"
git sparse-checkout init --cone
git sparse-checkout set dashboard/ reports/release/
git checkout main 2>/dev/null || git checkout master 2>/dev/null

echo ""
echo "==> Done! Contents:"
ls -la dashboard/dashboard.json dashboard/index.html 2>/dev/null
echo ""
echo "==> Generate dashboard (opens via file://, no server needed):"
echo "    cd $DIR && python3 dashboard/dashboard.py --release"
echo "    open $DIR/dashboard/index.html"
echo ""
echo "==> Alternative via HTTP server:"
echo "    cd $DIR && python3 -m http.server 8765 --directory ."
echo "    open http://127.0.0.1:8765/dashboard/index.html"
echo ""
echo "Note: dashboard.json is machine-readable data with profiler/correctness/batch_scaling."
echo "      Open index.html to view the dashboard."
