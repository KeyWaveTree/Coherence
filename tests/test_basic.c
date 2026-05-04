/**
 * Coherence - Basic Functionality Test
 *
 * 기본 API 동작 테스트
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cudabridge.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { \
    printf("[FAIL] %s: %s\n", name, msg); \
    failures++; \
} while(0)

static int failures = 0;

/* 초기화 테스트 */
void test_init(void)
{
    cbError_t err;

    /* 초기화 전 상태 확인 */
    if (cbIsInitialized()) {
        TEST_FAIL("pre-init check", "already initialized");
        return;
    }

    /* 초기화 */
    err = cbInit();
    if (err != cbSuccess) {
        TEST_FAIL("cbInit", cbGetErrorString(err));
        return;
    }

    /* 초기화 후 상태 확인 */
    if (!cbIsInitialized()) {
        TEST_FAIL("post-init check", "not initialized after cbInit");
        return;
    }

    TEST_PASS("initialization");
}

/* 버전 정보 테스트 */
void test_version(void)
{
    int version;
    cbError_t err;

    err = cbGetVersion(&version);
    if (err != cbSuccess) {
        TEST_FAIL("cbGetVersion", cbGetErrorString(err));
        return;
    }

    if (version < 10000) {
        TEST_FAIL("version check", "unexpected version");
        return;
    }

    printf("  Coherence version: %d.%d.%d\n",
           version / 10000, (version / 100) % 100, version % 100);

    err = cbGetDriverVersion(&version);
    if (err != cbSuccess) {
        TEST_FAIL("cbGetDriverVersion", cbGetErrorString(err));
        return;
    }

    printf("  Driver version: %d\n", version);

    TEST_PASS("version info");
}

/* 디바이스 열거 테스트 */
void test_device_enum(void)
{
    int count;
    cbError_t err;

    err = cbGetDeviceCount(&count);
    if (err != cbSuccess) {
        TEST_FAIL("cbGetDeviceCount", cbGetErrorString(err));
        return;
    }

    printf("  Found %d device(s)\n", count);

    if (count > 0) {
        int device;
        err = cbGetDevice(&device);
        if (err != cbSuccess) {
            TEST_FAIL("cbGetDevice", cbGetErrorString(err));
            return;
        }

        printf("  Current device: %d\n", device);

        err = cbSetDevice(0);
        if (err != cbSuccess) {
            TEST_FAIL("cbSetDevice", cbGetErrorString(err));
            return;
        }
    }

    TEST_PASS("device enumeration");
}

/* 디바이스 속성 테스트 */
void test_device_properties(void)
{
    int count;
    cbGetDeviceCount(&count);

    if (count == 0) {
        printf("  [SKIP] No devices available\n");
        return;
    }

    cbDeviceProp prop;
    cbError_t err = cbGetDeviceProperties(&prop, 0);
    if (err != cbSuccess) {
        TEST_FAIL("cbGetDeviceProperties", cbGetErrorString(err));
        return;
    }

    printf("  Device 0: %s\n", prop.name);
    printf("    Total memory: %.2f GB\n",
           (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("    Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("    Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("    Warp size: %d\n", prop.warpSize);

    TEST_PASS("device properties");
}

/* 메모리 할당 테스트 */
void test_memory_alloc(void)
{
    void *ptr = NULL;
    size_t size = 1024 * 1024;  /* 1 MB */
    cbError_t err;

    /* 디바이스 메모리 할당 */
    err = cbMalloc(&ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("cbMalloc", cbGetErrorString(err));
        return;
    }

    if (ptr == NULL) {
        TEST_FAIL("cbMalloc", "returned NULL pointer");
        return;
    }

    printf("  Allocated %zu bytes at %p\n", size, ptr);

    /* 메모리 해제 */
    err = cbFree(ptr);
    if (err != cbSuccess) {
        TEST_FAIL("cbFree", cbGetErrorString(err));
        return;
    }

    TEST_PASS("memory allocation");
}

/* 호스트 메모리 할당 테스트 */
void test_host_memory(void)
{
    void *ptr = NULL;
    size_t size = 1024 * 1024;
    cbError_t err;

    err = cbMallocHost(&ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("cbMallocHost", cbGetErrorString(err));
        return;
    }

    if (ptr == NULL) {
        TEST_FAIL("cbMallocHost", "returned NULL pointer");
        return;
    }

    /* 실제로 접근 가능한지 확인 */
    memset(ptr, 0xAB, size);

    err = cbFreeHost(ptr);
    if (err != cbSuccess) {
        TEST_FAIL("cbFreeHost", cbGetErrorString(err));
        return;
    }

    TEST_PASS("host memory allocation");
}

/* 메모리 복사 테스트 */
void test_memcpy(void)
{
    size_t size = 1024;
    void *h_src = malloc(size);
    void *h_dst = malloc(size);
    void *d_ptr = NULL;
    cbError_t err;

    /* 소스 데이터 초기화 */
    memset(h_src, 0x42, size);
    memset(h_dst, 0, size);

    /* 디바이스 메모리 할당 */
    err = cbMalloc(&d_ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("memcpy setup", cbGetErrorString(err));
        goto cleanup;
    }

    /* H2D 복사 */
    err = cbMemcpy(d_ptr, h_src, size, CB_MEMCPY_HOST_TO_DEVICE);
    if (err != cbSuccess) {
        TEST_FAIL("cbMemcpy H2D", cbGetErrorString(err));
        goto cleanup;
    }

    /* D2H 복사 */
    err = cbMemcpy(h_dst, d_ptr, size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        TEST_FAIL("cbMemcpy D2H", cbGetErrorString(err));
        goto cleanup;
    }

    /* 데이터 확인 */
    if (memcmp(h_src, h_dst, size) != 0) {
        TEST_FAIL("memcpy verify", "data mismatch");
        goto cleanup;
    }

    TEST_PASS("memory copy");

cleanup:
    if (d_ptr) cbFree(d_ptr);
    free(h_src);
    free(h_dst);
}

/* 메모리 설정 테스트 */
void test_memset(void)
{
    size_t size = 1024;
    void *d_ptr = NULL;
    void *h_ptr = malloc(size);
    cbError_t err;

    err = cbMalloc(&d_ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("memset setup", cbGetErrorString(err));
        goto cleanup;
    }

    err = cbMemset(d_ptr, 0xFF, size);
    if (err != cbSuccess) {
        TEST_FAIL("cbMemset", cbGetErrorString(err));
        goto cleanup;
    }

    err = cbMemcpy(h_ptr, d_ptr, size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        TEST_FAIL("memset verify copy", cbGetErrorString(err));
        goto cleanup;
    }

    /* 데이터 확인 */
    unsigned char *data = (unsigned char*)h_ptr;
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0xFF) {
            TEST_FAIL("memset verify", "data mismatch");
            goto cleanup;
        }
    }

    TEST_PASS("memory set");

cleanup:
    if (d_ptr) cbFree(d_ptr);
    free(h_ptr);
}

/* eGPU 연결 정보 테스트 */
void test_connection_info(void)
{
    cbConnectionInfo info;
    cbError_t err;

    err = cbGetConnectionInfo(&info);
    if (err != cbSuccess) {
        TEST_FAIL("cbGetConnectionInfo", cbGetErrorString(err));
        return;
    }

    printf("  Connection status: %s\n",
           info.isConnected ? "connected" : "not connected");

    if (info.isConnected) {
        printf("  Upstream bandwidth: %.2f GB/s\n",
               (double)info.upstreamBandwidth / (1024 * 1024 * 1024));
        printf("  Downstream bandwidth: %.2f GB/s\n",
               (double)info.downstreamBandwidth / (1024 * 1024 * 1024));
        printf("  PCIe Gen%d x%d\n", info.pcieGeneration, info.pcieLanes);
    }

    TEST_PASS("connection info");
}

/* 에러 처리 테스트 */
void test_error_handling(void)
{
    cbError_t err;

    /* 의도적으로 에러 발생 */
    err = cbMalloc(NULL, 0);
    if (err == cbSuccess) {
        TEST_FAIL("error handling", "expected error but got success");
        return;
    }

    /* 에러 이름 확인 */
    const char *name = cbGetErrorName(err);
    if (name == NULL || strlen(name) == 0) {
        TEST_FAIL("cbGetErrorName", "returned empty string");
        return;
    }

    /* 에러 설명 확인 */
    const char *str = cbGetErrorString(err);
    if (str == NULL || strlen(str) == 0) {
        TEST_FAIL("cbGetErrorString", "returned empty string");
        return;
    }

    printf("  Error name: %s\n", name);
    printf("  Error string: %s\n", str);

    /* 마지막 에러 확인 */
    err = cbGetLastError();
    /* 에러가 리셋되었는지 확인 */
    if (cbPeekAtLastError() != cbSuccess) {
        TEST_FAIL("cbGetLastError", "error not cleared");
        return;
    }

    TEST_PASS("error handling");
}

/* 종료 테스트 */
void test_shutdown(void)
{
    cbError_t err = cbShutdown();
    if (err != cbSuccess) {
        TEST_FAIL("cbShutdown", cbGetErrorString(err));
        return;
    }

    if (cbIsInitialized()) {
        TEST_FAIL("post-shutdown check", "still initialized after shutdown");
        return;
    }

    TEST_PASS("shutdown");
}

int main(void)
{
    printf("\n=== Coherence Basic Functionality Tests ===\n\n");

    test_init();
    test_version();
    test_device_enum();
    test_device_properties();
    test_memory_alloc();
    test_host_memory();
    test_memcpy();
    test_memset();
    test_connection_info();
    test_error_handling();
    test_shutdown();

    printf("\n=== Test Results ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
