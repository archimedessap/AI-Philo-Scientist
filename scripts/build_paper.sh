#!/usr/bin/env bash
set -euo pipefail

# Wrapper to build the paper PDF from repository root

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT_DIR/paper/build.sh"

