/**
 * Coherence Example - Vector Addition
 *
 * GPU를 사용한 벡터 덧셈 예제
 * 실제 CUDA 커널 대신 CPU 에뮬레이션을 사용합니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cudabridge.h>

#define N (1024 * 1024)  /* 1M 원소 */

/* 벡터 덧셈을 CPU에서 에뮬레이션 (실제 CUDA 커널 대용) */
void vector_add_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* 결과 검증 */
int verify_result(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: expected %f, got %f\n",
                   i, expected, c[i]);
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    printf("\n=== Coherence Vector Addition Example ===\n\n");

    /* Coherence 초기화 */
    cbError_t err = cbInit();
    if (err != cbSuccess) {
        printf("Failed to initialize Coherence: %s\n", cbGetErrorString(err));
        return 1;
    }

    /* 디바이스 정보 출력 */
    int device_count;
    cbGetDeviceCount(&device_count);
    printf("Found %d GPU(s)\n", device_count);

    if (device_count > 0) {
        cbDeviceProp prop;
        cbGetDeviceProperties(&prop, 0);
        printf("Using: %s\n", prop.name);
    }

    printf("\nVector size: %d elements (%.2f MB)\n",
           N, (float)(N * sizeof(float)) / (1024 * 1024));

    /* 호스트 메모리 할당 */
    float *h_a = NULL, *h_b = NULL, *h_c = NULL;

    err = cbMallocHost((void**)&h_a, N * sizeof(float));
    if (err != cbSuccess) goto cleanup;

    err = cbMallocHost((void**)&h_b, N * sizeof(float));
    if (err != cbSuccess) goto cleanup;

    err = cbMallocHost((void**)&h_c, N * sizeof(float));
    if (err != cbSuccess) goto cleanup;

    /* 입력 데이터 초기화 */
    printf("\nInitializing input data...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* 디바이스 메모리 할당 */
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;

    err = cbMalloc((void**)&d_a, N * sizeof(float));
    if (err != cbSuccess) {
        printf("Failed to allocate d_a: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    err = cbMalloc((void**)&d_b, N * sizeof(float));
    if (err != cbSuccess) {
        printf("Failed to allocate d_b: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    err = cbMalloc((void**)&d_c, N * sizeof(float));
    if (err != cbSuccess) {
        printf("Failed to allocate d_c: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    /* 이벤트 생성 (타이밍용) */
    cbEvent_t start, stop;
    cbEventCreate(&start);
    cbEventCreate(&stop);

    /* H2D 데이터 전송 */
    printf("Copying data to device...\n");

    cbEventRecord(start, NULL);

    err = cbMemcpy(d_a, h_a, N * sizeof(float), CB_MEMCPY_HOST_TO_DEVICE);
    if (err != cbSuccess) {
        printf("Failed to copy d_a: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    err = cbMemcpy(d_b, h_b, N * sizeof(float), CB_MEMCPY_HOST_TO_DEVICE);
    if (err != cbSuccess) {
        printf("Failed to copy d_b: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    cbEventRecord(stop, NULL);
    cbEventSynchronize(stop);

    float h2d_time;
    cbEventElapsedTime(&h2d_time, start, stop);
    printf("  H2D transfer time: %.2f ms (%.2f GB/s)\n",
           h2d_time,
           (2.0 * N * sizeof(float)) / (h2d_time / 1000.0) / (1024 * 1024 * 1024));

    /* 커널 실행 (CPU 에뮬레이션) */
    printf("Executing vector addition...\n");

    cbEventRecord(start, NULL);

    /* 실제 CUDA에서는 여기서 커널을 실행하지만,
     * 현재 구현에서는 CPU에서 에뮬레이션 */
    /* 데이터를 다시 호스트로 가져와서 계산 후 업로드 */
    cbMemcpy(h_a, d_a, N * sizeof(float), CB_MEMCPY_DEVICE_TO_HOST);
    cbMemcpy(h_b, d_b, N * sizeof(float), CB_MEMCPY_DEVICE_TO_HOST);

    vector_add_cpu(h_a, h_b, h_c, N);

    cbMemcpy(d_c, h_c, N * sizeof(float), CB_MEMCPY_HOST_TO_DEVICE);

    cbEventRecord(stop, NULL);
    cbEventSynchronize(stop);

    float kernel_time;
    cbEventElapsedTime(&kernel_time, start, stop);
    printf("  Kernel time: %.2f ms\n", kernel_time);

    /* D2H 결과 전송 */
    printf("Copying result to host...\n");

    cbEventRecord(start, NULL);

    err = cbMemcpy(h_c, d_c, N * sizeof(float), CB_MEMCPY_DEVICE_TO_HOST);
    if (err != cbSuccess) {
        printf("Failed to copy result: %s\n", cbGetErrorString(err));
        goto cleanup;
    }

    cbEventRecord(stop, NULL);
    cbEventSynchronize(stop);

    float d2h_time;
    cbEventElapsedTime(&d2h_time, start, stop);
    printf("  D2H transfer time: %.2f ms (%.2f GB/s)\n",
           d2h_time,
           (N * sizeof(float)) / (d2h_time / 1000.0) / (1024 * 1024 * 1024));

    /* 결과 검증 */
    printf("\nVerifying result...\n");

    /* 원본 데이터 복원 (검증용) */
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    if (verify_result(h_a, h_b, h_c, N)) {
        printf("Result: PASSED\n");
    } else {
        printf("Result: FAILED\n");
    }

    /* 총 시간 출력 */
    float total_time = h2d_time + kernel_time + d2h_time;
    printf("\nTotal time: %.2f ms\n", total_time);

    /* 샘플 결과 출력 */
    printf("\nSample results:\n");
    printf("  a[0] + b[0] = %f + %f = %f\n", h_a[0], h_b[0], h_c[0]);
    printf("  a[N/2] + b[N/2] = %f + %f = %f\n",
           h_a[N/2], h_b[N/2], h_c[N/2]);
    printf("  a[N-1] + b[N-1] = %f + %f = %f\n",
           h_a[N-1], h_b[N-1], h_c[N-1]);

cleanup:
    printf("\nCleaning up...\n");

    /* 이벤트 제거 */
    cbEventDestroy(start);
    cbEventDestroy(stop);

    /* 디바이스 메모리 해제 */
    if (d_a) cbFree(d_a);
    if (d_b) cbFree(d_b);
    if (d_c) cbFree(d_c);

    /* 호스트 메모리 해제 */
    if (h_a) cbFreeHost(h_a);
    if (h_b) cbFreeHost(h_b);
    if (h_c) cbFreeHost(h_c);

    /* Coherence 종료 */
    cbShutdown();

    printf("Done.\n\n");

    return 0;
}
