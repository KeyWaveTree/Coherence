/**
 * Coherence Example - Device Information
 *
 * 연결된 GPU 디바이스의 상세 정보를 출력합니다.
 */

#include <stdio.h>
#include <cudabridge.h>

void print_separator(void)
{
    printf("═══════════════════════════════════════════════════════════════\n");
}

void print_device_info(int device)
{
    cbDeviceProp prop;
    cbError_t err = cbGetDeviceProperties(&prop, device);

    if (err != cbSuccess) {
        printf("Failed to get properties for device %d: %s\n",
               device, cbGetErrorString(err));
        return;
    }

    printf("\n");
    print_separator();
    printf(" Device %d: %s\n", device, prop.name);
    print_separator();

    printf("\n【Compute Capability】\n");
    printf("  Version: %d.%d\n", prop.major, prop.minor);

    printf("\n【Memory】\n");
    printf("  Total Global Memory:    %.2f GB\n",
           (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Memory Clock Rate:      %d MHz\n", prop.memoryClockRate / 1000);
    printf("  Memory Bus Width:       %d bits\n", prop.memoryBusWidth);
    printf("  L2 Cache Size:          %d KB\n", prop.l2CacheSize / 1024);

    /* 메모리 대역폭 계산 */
    double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6;
    printf("  Peak Memory Bandwidth:  %.1f GB/s\n", bandwidth);

    printf("\n【Multiprocessor】\n");
    printf("  SM Count:               %d\n", prop.multiProcessorCount);
    printf("  Max Threads/SM:         %d\n", prop.maxThreadsPerMultiProcessor);

    /* CUDA 코어 수 추정 (아키텍처에 따라 다름) */
    int cores_per_sm = 128;  /* Ampere/Ada 기준 */
    if (prop.major == 7) cores_per_sm = 64;  /* Volta/Turing */
    else if (prop.major == 6) cores_per_sm = 128;  /* Pascal */
    int total_cores = prop.multiProcessorCount * cores_per_sm;
    printf("  CUDA Cores (est.):      %d\n", total_cores);

    printf("\n【Thread Configuration】\n");
    printf("  Warp Size:              %d\n", prop.warpSize);
    printf("  Max Threads/Block:      %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dimensions:   (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions:    (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("\n【Shared Memory】\n");
    printf("  Shared Memory/Block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Registers/Block:        %d\n", prop.regsPerBlock);

    printf("\n【Clock】\n");
    printf("  GPU Clock Rate:         %d MHz\n", prop.clockRate / 1000);

    printf("\n【Features】\n");
    printf("  Concurrent Kernels:     %s\n",
           prop.concurrentKernels ? "Yes" : "No");
    printf("  Async Engine Count:     %d\n", prop.asyncEngineCount);
    printf("  Unified Addressing:     %s\n",
           prop.unifiedAddressing ? "Yes" : "No");
    printf("  Managed Memory:         %s\n",
           prop.managedMemory ? "Yes" : "No");

    printf("\n");
}

void print_connection_info(void)
{
    cbConnectionInfo info;
    cbError_t err = cbGetConnectionInfo(&info);

    if (err != cbSuccess) {
        printf("Failed to get connection info: %s\n", cbGetErrorString(err));
        return;
    }

    print_separator();
    printf(" eGPU Connection Status\n");
    print_separator();

    printf("\n【Connection】\n");
    printf("  Status:                 %s\n",
           info.isConnected ? "Connected" : "Not Connected");

    if (info.isConnected) {
        const char *conn_type;
        switch (info.connectionType) {
            case 0: conn_type = "USB4"; break;
            case 1: conn_type = "Thunderbolt 3"; break;
            case 2: conn_type = "Thunderbolt 4"; break;
            default: conn_type = "Unknown"; break;
        }
        printf("  Connection Type:        %s\n", conn_type);
        printf("  PCIe Generation:        Gen %d\n", info.pcieGeneration);
        printf("  PCIe Lanes:             x%d\n", info.pcieLanes);

        printf("\n【Bandwidth】\n");
        printf("  Upstream (H→D):         %.2f GB/s\n",
               (double)info.upstreamBandwidth / (1024 * 1024 * 1024));
        printf("  Downstream (D→H):       %.2f GB/s\n",
               (double)info.downstreamBandwidth / (1024 * 1024 * 1024));
        printf("  Link Utilization:       %.1f%%\n",
               info.linkUtilization * 100);
    }

    printf("\n");
}

void print_memory_info(void)
{
    size_t free, total;
    cbError_t err = cbMemGetInfo(&free, &total);

    if (err != cbSuccess) {
        printf("Failed to get memory info: %s\n", cbGetErrorString(err));
        return;
    }

    print_separator();
    printf(" Memory Status\n");
    print_separator();

    printf("\n");
    printf("  Total:     %.2f GB\n", (double)total / (1024 * 1024 * 1024));
    printf("  Free:      %.2f GB\n", (double)free / (1024 * 1024 * 1024));
    printf("  Used:      %.2f GB (%.1f%%)\n",
           (double)(total - free) / (1024 * 1024 * 1024),
           100.0 * (total - free) / total);
    printf("\n");
}

int main(void)
{
    printf("\n");
    print_separator();
    printf(" Coherence - GPU Device Information\n");
    print_separator();

    /* Coherence 초기화 */
    cbError_t err = cbInit();
    if (err != cbSuccess) {
        printf("\nFailed to initialize Coherence: %s\n", cbGetErrorString(err));
        return 1;
    }

    /* 버전 정보 */
    int version;
    cbGetVersion(&version);
    printf("\n  Coherence Version: %d.%d.%d\n",
           version / 10000, (version / 100) % 100, version % 100);

    cbGetDriverVersion(&version);
    printf("  CUDA Driver Version: %d.%d\n",
           version / 1000, (version % 1000) / 10);

    /* 디바이스 수 */
    int device_count;
    err = cbGetDeviceCount(&device_count);
    if (err != cbSuccess) {
        printf("\nFailed to get device count: %s\n", cbGetErrorString(err));
        cbShutdown();
        return 1;
    }

    printf("  Available Devices: %d\n", device_count);

    /* 연결 정보 */
    print_connection_info();

    /* 각 디바이스 정보 출력 */
    for (int i = 0; i < device_count; i++) {
        cbSetDevice(i);
        print_device_info(i);
    }

    /* 메모리 정보 */
    if (device_count > 0) {
        cbSetDevice(0);
        print_memory_info();
    }

    /* 종료 */
    cbShutdown();

    return 0;
}
