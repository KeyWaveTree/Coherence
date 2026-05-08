#!/bin/bash
# Coherence - Uninstallation Script
set -e

PREFIX="${INSTALL_PREFIX:-/usr/local}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[Coherence]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $*"; }

log "Coherence Uninstaller"
echo "============================================"
echo ""

NEED_SUDO=false
if [ "$PREFIX" = "/usr/local" ] || [ "$PREFIX" = "/usr" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        NEED_SUDO=true
    fi
fi

do_rm() {
    if [ -e "$1" ]; then
        if [ "$NEED_SUDO" = true ]; then
            sudo rm -rf "$1"
        else
            rm -rf "$1"
        fi
        log "  Removed: $1"
    fi
}

log "Removing Coherence files from ${PREFIX}..."
echo ""

# 헤더
do_rm "${PREFIX}/include/cudabridge.h"

# 라이브러리
do_rm "${PREFIX}/lib/libcudabridge.so"
do_rm "${PREFIX}/lib/libcudabridge.so.1"
do_rm "${PREFIX}/lib/libcudabridge.so.1.0.0"
do_rm "${PREFIX}/lib/libcudabridge.a"
do_rm "${PREFIX}/lib/libcudabridge.dylib"
do_rm "${PREFIX}/lib/libcudabridge.1.dylib"
do_rm "${PREFIX}/lib/libcudabridge.1.0.0.dylib"

# pkg-config
do_rm "${PREFIX}/lib/pkgconfig/cudabridge.pc"

# CMake 설정
do_rm "${PREFIX}/lib/cmake/Coherence"

# ldconfig (Linux)
if [ "$(uname -s)" = "Linux" ]; then
    for conf in coherence.conf cudabridge.conf; do
        if [ -f "/etc/ld.so.conf.d/$conf" ]; then
            if [ "$NEED_SUDO" = true ]; then
                sudo rm -f "/etc/ld.so.conf.d/$conf"
            else
                rm -f "/etc/ld.so.conf.d/$conf"
            fi
            log "  Removed ldconfig entry: $conf"
        fi
    done
    ldconfig 2>/dev/null || sudo ldconfig 2>/dev/null || true
fi

echo ""
log "Coherence uninstalled successfully."
