#!/bin/bash
# Coherence - Installation Script
# Apple Silicon Mac에서 eGPU CUDA 드라이버 설치
set -e

VERSION="1.0.0"
PREFIX="${INSTALL_PREFIX:-/usr/local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="$(mktemp -d)"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[Coherence]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# 루트 디렉토리 탐지
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

log "Coherence v${VERSION} Installer"
echo "============================================"
echo ""

# 시스템 확인
log "Checking system requirements..."

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" != "arm64" ]; then
        warn "Coherence is designed for Apple Silicon (arm64). Detected: $ARCH"
        warn "Building anyway, but eGPU features require Apple Silicon."
    fi

    # macOS 버전 확인
    MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
    log "macOS version: $MACOS_VER"
    log "Architecture: $ARCH"

    # Xcode 커맨드 라인 도구 확인
    if ! xcode-select -p &>/dev/null; then
        error "Xcode Command Line Tools required. Install with: xcode-select --install"
    fi
elif [ "$OS" = "Linux" ]; then
    warn "Building on Linux. macOS-specific features will use simulation mode."
    log "Architecture: $ARCH"
else
    error "Unsupported OS: $OS"
fi

# cmake 확인
if ! command -v cmake &>/dev/null; then
    error "CMake is required. Install with: brew install cmake (macOS) or apt install cmake (Linux)"
fi

CMAKE_VER="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+')"
log "CMake version: $CMAKE_VER"

# 빌드
log "Building Coherence (${BUILD_TYPE})..."
echo ""

cd "$BUILD_DIR"
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=ON

NPROC=4
if command -v nproc &>/dev/null; then
    NPROC="$(nproc)"
elif command -v sysctl &>/dev/null; then
    NPROC="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

cmake --build . --parallel "$NPROC"

# 테스트 실행
log "Running tests..."
echo ""

TESTS_PASSED=true
for test_bin in test_basic test_memory test_stream; do
    if [ -f "./$test_bin" ]; then
        log "  Running $test_bin..."
        if ./"$test_bin" > /dev/null 2>&1; then
            echo -e "    ${GREEN}PASSED${NC}"
        else
            echo -e "    ${RED}FAILED${NC}"
            TESTS_PASSED=false
        fi
    fi
done

if [ "$TESTS_PASSED" = false ]; then
    error "Tests failed. Aborting installation."
fi

echo ""
log "All tests passed."

# 설치
log "Installing to ${PREFIX}..."
echo ""

if [ "$PREFIX" = "/usr/local" ] || [ "$PREFIX" = "/usr" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        log "Root privileges required for installation to $PREFIX"
        sudo cmake --install .
    else
        cmake --install .
    fi
else
    cmake --install .
fi

# ldconfig (Linux)
if [ "$OS" = "Linux" ]; then
    if [ -f /etc/ld.so.conf ]; then
        if ! grep -q "$PREFIX/lib" /etc/ld.so.conf.d/*.conf 2>/dev/null; then
            log "Updating library cache..."
            echo "$PREFIX/lib" | sudo tee /etc/ld.so.conf.d/cudabridge.conf > /dev/null
            sudo ldconfig
        fi
    fi
fi

echo ""
echo "============================================"
log "Coherence v${VERSION} installed successfully!"
echo ""
echo "  Headers: ${PREFIX}/include/cudabridge.h"
echo "  Library: ${PREFIX}/lib/libcudabridge.so (shared)"
echo "           ${PREFIX}/lib/libcudabridge.a  (static)"
echo "  Pkg-config: ${PREFIX}/lib/pkgconfig/cudabridge.pc"
echo ""
echo "Usage:"
echo "  #include <cudabridge.h>"
echo ""
echo "  Compile: gcc -o myapp myapp.c -lcudabridge"
echo "  Or:      gcc -o myapp myapp.c \$(pkg-config --cflags --libs cudabridge)"
echo ""
echo "  CMake:   find_package(Coherence REQUIRED)"
echo "           target_link_libraries(myapp Coherence::cudabridge)"
echo ""
