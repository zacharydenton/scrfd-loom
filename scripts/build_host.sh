#!/usr/bin/env bash
# Build the command-line tools and the resident C ABI used by scrfd_loom.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

hipcc="${HIPCC:-/opt/rocm/bin/hipcc}"
mkdir -p build

"$hipcc" -O2 -Wall -Werror -fPIC -shared -DSCRFD_LIBRARY -o build/libscrfd.so host/scrfd.cpp
"$hipcc" -O2 -Wall -Werror -o host/scrfd host/scrfd.cpp
"$hipcc" -O2 -Wall -Werror -o host/loomrun host/loomrun.cpp

printf 'built %s\n' build/libscrfd.so host/scrfd host/loomrun
