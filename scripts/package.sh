#!/bin/bash
# Coherence - Distribution Packaging Script
# 배포 가능한 아카이브 및 macOS pkg 생성
set -e

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$PROJECT_DIR/dist"
BUILD_DIR="$(mktemp -d)"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[Package]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

OS="$(uname -s)"
ARCH="$(uname -m)"
PLATFORM="${OS}-${ARCH}"

log "Coherence v${VERSION} Packaging"
echo "============================================"
log "Platform: ${PLATFORM}"
echo ""

# dist 디렉토리
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# =============================================
# 1. Release 빌드
# =============================================
log "Building Release..."
STAGE_DIR="$BUILD_DIR/stage"
INSTALL_PREFIX="$STAGE_DIR/usr/local"

mkdir -p "$BUILD_DIR/release"
cd "$BUILD_DIR/release"

cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
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
log "Running tests before packaging..."
TESTS_OK=true
for t in test_basic test_memory test_stream; do
    if [ -f "./$t" ]; then
        if ./"$t" > /dev/null 2>&1; then
            log "  $t: PASSED"
        else
            log "  $t: FAILED"
            TESTS_OK=false
        fi
    fi
done

if [ "$TESTS_OK" = false ]; then
    error "Tests failed. Cannot package."
fi

# install to staging
cmake --install .

# =============================================
# 2. tar.gz 아카이브 생성
# =============================================
log "Creating distribution archive..."

ARCHIVE_NAME="cudabridge-${VERSION}-${PLATFORM}"
ARCHIVE_DIR="$BUILD_DIR/${ARCHIVE_NAME}"
mkdir -p "$ARCHIVE_DIR"

# 바이너리 배포 구조
mkdir -p "$ARCHIVE_DIR"/{lib,include,share/doc/cudabridge,share/pkgconfig,examples,bin}

# 라이브러리 복사
cp -a "$INSTALL_PREFIX"/lib/libcudabridge* "$ARCHIVE_DIR/lib/" 2>/dev/null || true

# 헤더 복사
cp "$INSTALL_PREFIX/include/cudabridge.h" "$ARCHIVE_DIR/include/"

# CUDA 호환 헤더 복사
if [ -f "$PROJECT_DIR/src/cuda/runtime/cuda_runtime.h" ]; then
    mkdir -p "$ARCHIVE_DIR/include/cuda"
    cp "$PROJECT_DIR/src/cuda/runtime/cuda_runtime.h" "$ARCHIVE_DIR/include/cuda/"
fi

# pkg-config
cp "$INSTALL_PREFIX"/lib/pkgconfig/cudabridge.pc "$ARCHIVE_DIR/share/pkgconfig/" 2>/dev/null || true

# 문서
cp "$PROJECT_DIR/LICENSE" "$ARCHIVE_DIR/share/doc/cudabridge/" 2>/dev/null || true
cp "$PROJECT_DIR/README.md" "$ARCHIVE_DIR/share/doc/cudabridge/" 2>/dev/null || true

# 예제 소스
cp "$PROJECT_DIR"/src/userspace/examples/*.c "$ARCHIVE_DIR/examples/" 2>/dev/null || true

# 예제 바이너리
for ex in example_vector_add example_device_info example_bandwidth; do
    if [ -f "$BUILD_DIR/release/$ex" ]; then
        cp "$BUILD_DIR/release/$ex" "$ARCHIVE_DIR/bin/"
    fi
done

# 설치/제거 스크립트
cp "$PROJECT_DIR/scripts/install.sh" "$ARCHIVE_DIR/"
cp "$PROJECT_DIR/scripts/uninstall.sh" "$ARCHIVE_DIR/"
chmod +x "$ARCHIVE_DIR/install.sh" "$ARCHIVE_DIR/uninstall.sh"

# README for binary package
cat > "$ARCHIVE_DIR/INSTALL.txt" << 'INSTALLEOF'
Coherence - Installation Guide
================================

Quick Install (from source):
  ./install.sh

Manual Install (binary):
  sudo cp lib/libcudabridge.* /usr/local/lib/
  sudo cp include/cudabridge.h /usr/local/include/
  sudo cp include/cuda/cuda_runtime.h /usr/local/include/cuda/  # optional
  sudo cp share/pkgconfig/cudabridge.pc /usr/local/lib/pkgconfig/

  # macOS:
  sudo update_dyld_shared_cache

  # Linux:
  sudo ldconfig

Uninstall:
  ./uninstall.sh

Usage:
  #include <cudabridge.h>

  cbInit();
  // ... your CUDA code ...
  cbShutdown();

Compile:
  gcc -o myapp myapp.c -lcudabridge
  # or
  gcc -o myapp myapp.c $(pkg-config --cflags --libs cudabridge)

Requirements:
  - macOS 13+ with Apple Silicon (M1/M2/M3/M4) for eGPU support
  - USB4/Thunderbolt 4 port
  - NVIDIA eGPU enclosure with supported GPU
  - Linux x86_64/arm64 also supported (simulation mode)
INSTALLEOF

# tar.gz 생성
cd "$BUILD_DIR"
tar czf "$DIST_DIR/${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
log "Created: dist/${ARCHIVE_NAME}.tar.gz"

# =============================================
# 3. macOS .pkg 생성 (macOS에서만)
# =============================================
if [ "$OS" = "Darwin" ] && command -v pkgbuild &>/dev/null; then
    log "Creating macOS installer package..."

    PKG_NAME="Coherence-${VERSION}.pkg"

    pkgbuild \
        --root "$STAGE_DIR" \
        --identifier "com.cudabridge.driver" \
        --version "$VERSION" \
        --install-location "/" \
        "$DIST_DIR/$PKG_NAME"

    log "Created: dist/$PKG_NAME"
fi

# =============================================
# 4. 소스 아카이브 생성
# =============================================
log "Creating source archive..."

cd "$PROJECT_DIR"
SRC_ARCHIVE="cudabridge-${VERSION}-source"

# git archive 사용 (있으면)
if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null; then
    git archive --format=tar.gz --prefix="${SRC_ARCHIVE}/" -o "$DIST_DIR/${SRC_ARCHIVE}.tar.gz" HEAD
else
    # git 없으면 수동 생성
    cd "$PROJECT_DIR/.."
    tar czf "$DIST_DIR/${SRC_ARCHIVE}.tar.gz" \
        --exclude='build' \
        --exclude='dist' \
        --exclude='.git' \
        --transform "s/^Coherence/${SRC_ARCHIVE}/" \
        Coherence/
fi

log "Created: dist/${SRC_ARCHIVE}.tar.gz"

# =============================================
# 5. 체크섬 생성
# =============================================
log "Generating checksums..."

cd "$DIST_DIR"
if command -v sha256sum &>/dev/null; then
    sha256sum *.tar.gz *.pkg 2>/dev/null > SHA256SUMS.txt || sha256sum *.tar.gz > SHA256SUMS.txt
elif command -v shasum &>/dev/null; then
    shasum -a 256 *.tar.gz *.pkg 2>/dev/null > SHA256SUMS.txt || shasum -a 256 *.tar.gz > SHA256SUMS.txt
fi

log "Created: dist/SHA256SUMS.txt"

# =============================================
# 요약
# =============================================
echo ""
echo "============================================"
log "Packaging complete!"
echo ""
echo "Distribution files:"
ls -lh "$DIST_DIR"/ | grep -v '^total'
echo ""
echo "To install from binary package:"
echo "  tar xzf ${ARCHIVE_NAME}.tar.gz"
echo "  cd ${ARCHIVE_NAME}"
echo "  ./install.sh"
echo ""
if [ "$OS" = "Darwin" ]; then
    echo "Or install via macOS package:"
    echo "  sudo installer -pkg Coherence-${VERSION}.pkg -target /"
    echo ""
fi
echo "To install from source:"
echo "  tar xzf ${SRC_ARCHIVE}.tar.gz"
echo "  cd ${SRC_ARCHIVE}"
echo "  ./scripts/install.sh"
echo ""
