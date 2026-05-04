/**
 * Coherence - Memory Management Test
 *
 * 메모리 할당, 전송, 해제 테스트
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

/* 다중 할당 테스트 */
void test_multiple_alloc(void)
{
    const int NUM_ALLOCS = 100;
    void *ptrs[NUM_ALLOCS];
    size_t size = 4096;
    cbError_t err;

    printf("  Allocating %d blocks of %zu bytes each...\n", NUM_ALLOCS, size);

    /* 여러 번 할당 */
    for (int i = 0; i < NUM_ALLOCS; i++) {
        err = cbMalloc(&ptrs[i], size);
        if (err != cbSuccess) {
            TEST_FAIL("multiple alloc", cbGetErrorString(err));
            /* 이미 할당된 것들 정리 */
            for (int j = 0; j < i; j++) {
                cbFree(ptrs[j]);
            }
            return;
        }
    }

    /* 모두 해제 */
    for (int i = 0; i < NUM_ALLOCS; i++) {
        err = cbFree(ptrs[i]);
        if (err != cbSuccess) {
            TEST_FAIL("multiple free", cbGetErrorString(err));
            return;
        }
    }

    TEST_PASS("multiple allocation");
}

/* 대용량 할당 테스트 */
void test_large_alloc(void)
{
    size_t total, free_before, free_after;
    void *ptr = NULL;
    cbError_t err;

    err = cbMemGetInfo(&free_before, &total);
    if (err != cbSuccess) {
        TEST_FAIL("large alloc setup", cbGetErrorString(err));
        return;
    }

    printf("  Total memory: %.2f GB, Free: %.2f GB\n",
           (double)total / (1024 * 1024 * 1024),
           (double)free_before / (1024 * 1024 * 1024));

    /* 큰 메모리 할당 (사용 가능 메모리의 10%) */
    size_t alloc_size = free_before / 10;
    printf("  Allocating %.2f MB...\n", (double)alloc_size / (1024 * 1024));

    err = cbMalloc(&ptr, alloc_size);
    if (err != cbSuccess) {
        TEST_FAIL("large alloc", cbGetErrorString(err));
        return;
    }

    err = cbMemGetInfo(&free_after, &total);
    if (err != cbSuccess) {
        cbFree(ptr);
        TEST_FAIL("large alloc verify", cbGetErrorString(err));
        return;
    }

    printf("  Free memory after alloc: %.2f GB\n",
           (double)free_after / (1024 * 1024 * 1024));

    cbFree(ptr);

    TEST_PASS("large allocation");
}

/* 메모리 전송 성능 테스트 */
void test_transfer_performance(void)
{
    size_t size = 64 * 1024 * 1024;  /* 64 MB */
    void *h_data = NULL;
    void *d_data = NULL;
    cbEvent_t start, end;
    float elapsed;
    cbError_t err;

    /* 호스트 메모리 할당 */
    err = cbMallocHost(&h_data, size);
    if (err != cbSuccess) {
        TEST_FAIL("transfer perf setup", cbGetErrorString(err));
        return;
    }

    /* 디바이스 메모리 할당 */
    err = cbMalloc(&d_data, size);
    if (err != cbSuccess) {
        cbFreeHost(h_data);
        TEST_FAIL("transfer perf setup", cbGetErrorString(err));
        return;
    }

    /* 데이터 초기화 */
    memset(h_data, 0x42, size);

    /* 이벤트 생성 */
    cbEventCreate(&start);
    cbEventCreate(&end);

    /* H2D 전송 측정 */
    cbEventRecord(start, NULL);
    err = cbMemcpy(d_data, h_data, size, CB_MEMCPY_HOST_TO_DEVICE);
    cbEventRecord(end, NULL);
    cbEventSynchronize(end);

    if (err == cbSuccess) {
        cbEventElapsedTime(&elapsed, start, end);
        double bandwidth = (double)size / (elapsed / 1000.0) / (1024 * 1024 * 1024);
        printf("  H2D: %.2f ms (%.2f GB/s)\n", elapsed, bandwidth);
    }

    /* D2H 전송 측정 */
    cbEventRecord(start, NULL);
    err = cbMemcpy(h_data, d_data, size, CB_MEMCPY_DEVICE_TO_HOST);
    cbEventRecord(end, NULL);
    cbEventSynchronize(end);

    if (err == cbSuccess) {
        cbEventElapsedTime(&elapsed, start, end);
        double bandwidth = (double)size / (elapsed / 1000.0) / (1024 * 1024 * 1024);
        printf("  D2H: %.2f ms (%.2f GB/s)\n", elapsed, bandwidth);
    }

    /* 정리 */
    cbEventDestroy(start);
    cbEventDestroy(end);
    cbFree(d_data);
    cbFreeHost(h_data);

    TEST_PASS("transfer performance");
}

/* 통합 메모리 테스트 */
void test_managed_memory(void)
{
    size_t size = 4096;
    void *ptr = NULL;
    cbError_t err;

    err = cbMallocManaged(&ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("managed memory alloc", cbGetErrorString(err));
        return;
    }

    /* 호스트에서 접근 */
    memset(ptr, 0xAB, size);

    /* 동기화 */
    cbDeviceSynchronize();

    /* 데이터 확인 */
    unsigned char *data = (unsigned char*)ptr;
    int valid = 1;
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0xAB) {
            valid = 0;
            break;
        }
    }

    cbFree(ptr);  /* 통합 메모리도 cbFree로 해제 */

    if (valid) {
        TEST_PASS("managed memory");
    } else {
        TEST_FAIL("managed memory", "data verification failed");
    }
}

/* 패턴 전송 테스트 */
void test_pattern_transfer(void)
{
    size_t size = 1024;
    unsigned char *h_src = malloc(size);
    unsigned char *h_dst = malloc(size);
    void *d_ptr = NULL;
    cbError_t err;

    /* 패턴 생성 */
    for (size_t i = 0; i < size; i++) {
        h_src[i] = (unsigned char)(i & 0xFF);
    }
    memset(h_dst, 0, size);

    err = cbMalloc(&d_ptr, size);
    if (err != cbSuccess) {
        TEST_FAIL("pattern transfer setup", cbGetErrorString(err));
        goto cleanup;
    }

    /* H2D */
    err = cbMemcpy(d_ptr, h_src, size, CB_MEMCPY_HOST_TO_DEVICE);
    if (err != cbSuccess) {
        TEST_FAIL("pattern transfer H2D", cbGetErrorString(err));
        goto cleanup;
    }

    /* D2H */
    err = cbMemcpy(h_dst, d_ptr, size, CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        TEST_FAIL("pattern transfer D2H", cbGetErrorString(err));
        goto cleanup;
    }

    /* 패턴 확인 */
    for (size_t i = 0; i < size; i++) {
        if (h_src[i] != h_dst[i]) {
            char msg[64];
            snprintf(msg, sizeof(msg), "mismatch at offset %zu", i);
            TEST_FAIL("pattern transfer verify", msg);
            goto cleanup;
        }
    }

    TEST_PASS("pattern transfer");

cleanup:
    if (d_ptr) cbFree(d_ptr);
    free(h_src);
    free(h_dst);
}

int main(void)
{
    printf("\n=== Coherence Memory Management Tests ===\n\n");

    cbError_t err = cbInit();
    if (err != cbSuccess) {
        printf("Failed to initialize Coherence: %s\n", cbGetErrorString(err));
        return 1;
    }

    test_multiple_alloc();
    test_large_alloc();
    test_transfer_performance();
    test_managed_memory();
    test_pattern_transfer();

    cbShutdown();

    printf("\n=== Test Results ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
}
