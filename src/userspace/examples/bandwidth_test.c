/**
 * Coherence Example - Bandwidth Test
 *
 * USB4/Thunderbolt를 통한 eGPU 데이터 전송 대역폭을 측정합니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cudabridge.h>

/* 테스트 크기 배열 */
static size_t test_sizes[] = {
    1 * 1024,           /* 1 KB */
    4 * 1024,           /* 4 KB */
    16 * 1024,          /* 16 KB */
    64 * 1024,          /* 64 KB */
    256 * 1024,         /* 256 KB */
    1 * 1024 * 1024,    /* 1 MB */
    4 * 1024 * 1024,    /* 4 MB */
    16 * 1024 * 1024,   /* 16 MB */
    64 * 1024 * 1024,   /* 64 MB */
    256 * 1024 * 1024,  /* 256 MB */
};

#define NUM_SIZES (sizeof(test_sizes) / sizeof(test_sizes[0]))
#define NUM_ITERATIONS 10

/* 크기를 읽기 쉬운 형식으로 변환 */
void format_size(size_t bytes, char *buf, size_t buf_size)
{
    if (bytes >= 1024 * 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f GB", (double)bytes / (1024 * 1024 * 1024));
    } else if (bytes >= 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f MB", (double)bytes / (1024 * 1024));
    } else if (bytes >= 1024) {
        snprintf(buf, buf_size, "%.1f KB", (double)bytes / 1024);
    } else {
        snprintf(buf, buf_size, "%zu B", bytes);
    }
}

/* H2D 대역폭 측정 */
double measure_h2d_bandwidth(void *h_data, void *d_data, size_t size,
                             cbEvent_t start, cbEvent_t end)
{
    float min_time = 1e9;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cbEventRecord(start, NULL);
        cbMemcpy(d_data, h_data, size, CB_MEMCPY_HOST_TO_DEVICE);
        cbEventRecord(end, NULL);
        cbEventSynchronize(end);

        float elapsed;
        cbEventElapsedTime(&elapsed, start, end);
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }

    /* GB/s 계산 */
    return (double)size / (min_time / 1000.0) / (1024 * 1024 * 1024);
}

/* D2H 대역폭 측정 */
double measure_d2h_bandwidth(void *h_data, void *d_data, size_t size,
                             cbEvent_t start, cbEvent_t end)
{
    float min_time = 1e9;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cbEventRecord(start, NULL);
        cbMemcpy(h_data, d_data, size, CB_MEMCPY_DEVICE_TO_HOST);
        cbEventRecord(end, NULL);
        cbEventSynchronize(end);

        float elapsed;
        cbEventElapsedTime(&elapsed, start, end);
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }

    return (double)size / (min_time / 1000.0) / (1024 * 1024 * 1024);
}

void print_header(void)
{
    printf("\n");
    printf("┌──────────────┬──────────────────┬──────────────────┐\n");
    printf("│  Data Size   │   H2D (GB/s)     │   D2H (GB/s)     │\n");
    printf("├──────────────┼──────────────────┼──────────────────┤\n");
}

void print_row(const char *size_str, double h2d_bw, double d2h_bw)
{
    printf("│ %12s │ %14.2f   │ %14.2f   │\n", size_str, h2d_bw, d2h_bw);
}

void print_footer(void)
{
    printf("└──────────────┴──────────────────┴──────────────────┘\n");
}

int main(void)
{
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("     Coherence - USB4/Thunderbolt eGPU Bandwidth Test\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Coherence 초기화 */
    cbError_t err = cbInit();
    if (err != cbSuccess) {
        printf("Failed to initialize Coherence: %s\n", cbGetErrorString(err));
        return 1;
    }

    /* 디바이스 정보 */
    int device_count;
    cbGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("\nNo GPU devices found!\n");
        cbShutdown();
        return 1;
    }

    cbDeviceProp prop;
    cbGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    /* 연결 정보 */
    cbConnectionInfo conn;
    cbGetConnectionInfo(&conn);

    if (conn.isConnected) {
        printf("\nConnection: ");
        switch (conn.connectionType) {
            case 0: printf("USB4"); break;
            case 1: printf("Thunderbolt 3"); break;
            case 2: printf("Thunderbolt 4"); break;
            default: printf("Unknown"); break;
        }
        printf(" (PCIe Gen%d x%d)\n", conn.pcieGeneration, conn.pcieLanes);

        printf("Theoretical bandwidth: %.1f GB/s (per direction)\n",
               (double)conn.upstreamBandwidth / (1024 * 1024 * 1024));
    }

    /* 최대 크기의 메모리 할당 */
    size_t max_size = test_sizes[NUM_SIZES - 1];
    void *h_data = NULL;
    void *d_data = NULL;

    err = cbMallocHost(&h_data, max_size);
    if (err != cbSuccess) {
        printf("Failed to allocate host memory: %s\n", cbGetErrorString(err));
        cbShutdown();
        return 1;
    }

    err = cbMalloc(&d_data, max_size);
    if (err != cbSuccess) {
        printf("Failed to allocate device memory: %s\n", cbGetErrorString(err));
        cbFreeHost(h_data);
        cbShutdown();
        return 1;
    }

    /* 호스트 데이터 초기화 */
    memset(h_data, 0xAB, max_size);

    /* 이벤트 생성 */
    cbEvent_t start, end;
    cbEventCreate(&start);
    cbEventCreate(&end);

    /* 워밍업 */
    printf("\nWarming up...\n");
    cbMemcpy(d_data, h_data, 1024 * 1024, CB_MEMCPY_HOST_TO_DEVICE);
    cbMemcpy(h_data, d_data, 1024 * 1024, CB_MEMCPY_DEVICE_TO_HOST);
    cbDeviceSynchronize();

    /* 대역폭 측정 */
    printf("\nMeasuring bandwidth (%d iterations per size)...\n", NUM_ITERATIONS);

    print_header();

    double peak_h2d = 0, peak_d2h = 0;

    for (size_t i = 0; i < NUM_SIZES; i++) {
        size_t size = test_sizes[i];
        char size_str[32];
        format_size(size, size_str, sizeof(size_str));

        double h2d_bw = measure_h2d_bandwidth(h_data, d_data, size, start, end);
        double d2h_bw = measure_d2h_bandwidth(h_data, d_data, size, start, end);

        print_row(size_str, h2d_bw, d2h_bw);

        if (h2d_bw > peak_h2d) peak_h2d = h2d_bw;
        if (d2h_bw > peak_d2h) peak_d2h = d2h_bw;
    }

    print_footer();

    /* 결과 요약 */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("                         Summary\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Peak H2D Bandwidth: %.2f GB/s\n", peak_h2d);
    printf("  Peak D2H Bandwidth: %.2f GB/s\n", peak_d2h);

    if (conn.isConnected) {
        double theoretical = (double)conn.upstreamBandwidth / (1024 * 1024 * 1024);
        printf("\n  H2D Efficiency: %.1f%%\n", (peak_h2d / theoretical) * 100);
        printf("  D2H Efficiency: %.1f%%\n", (peak_d2h / theoretical) * 100);
    }

    printf("\n  Note: USB4 theoretical max is ~4 GB/s per direction\n");
    printf("        (32 Gbps with encoding overhead)\n");

    /* 정리 */
    cbEventDestroy(start);
    cbEventDestroy(end);
    cbFree(d_data);
    cbFreeHost(h_data);
    cbShutdown();

    printf("\n");

    return 0;
}
