#!/usr/bin/env bash
# Sparse checkout — download only dashboard/ + reports/release/
# Usage: ./download_dashboard.sh <repo-url>
#
# Then: python dashboard.py --release reports/release/current_release.json
#       Open dashboard/index.html in browser.

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
ls -la dashboard/ reports/release/current_release.json
echo ""
echo "==> Generate dashboard:"
echo "    cd $DIR && python dashboard/dashboard.py --release reports/release/current_release.json"
echo "    Open $DIR/dashboard/index.html in your browser."
