#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-generic}"

exec "$root_dir/scripts/build.sh" "$backend" test
