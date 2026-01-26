/**
 * CudaBridge - CUDA Runtime API Implementation
 *
 * CUDA Runtime API 호환 레이어 구현
 */

#include "cuda_runtime.h"
#include "../../kernel/usb4/usb4_controller.h"
#include "../../kernel/pcie/pcie_tunnel.h"
#include "../../driver/nvidia/nvidia_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* 로깅 매크로 */
#define CUDA_LOG(fmt, ...) printf("[CUDA] " fmt "\n", ##__VA_ARGS__)
#define CUDA_ERR_LOG(fmt, ...) fprintf(stderr, "[CUDA ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef DEBUG
#define CUDA_DBG(fmt, ...) printf("[CUDA DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define CUDA_DBG(fmt, ...)
#endif

/* 호스트/관리 메모리 추적 */
#define MAX_HOST_ALLOCS 1024

typedef struct HostAllocInfo {
    void *ptr;
    bool is_managed;
} HostAllocInfo;

/* 전역 상태 */
typedef struct CudaBridgeState {
    bool                    initialized;
    USB4ControllerContext   usb4_ctx;
    PCIeTunnelContext       pcie_ctx;
    NVGpuContext           *gpu_contexts[8];
    int                     gpu_count;
    int                     current_device;
    cudaError_t             last_error;
    pthread_mutex_t         lock;
    HostAllocInfo           host_allocs[MAX_HOST_ALLOCS];
    int                     host_alloc_count;
} CudaBridgeState;

static CudaBridgeState g_state = { .initialized = false };

/* 스트림 구조체 */
struct CUstream_st {
    int device;
    NVChannel *channel;
    bool is_default;
};

/* 이벤트 구조체 */
struct CUevent_st {
    int device;
    uint64_t timestamp;
    bool recorded;
};

/* 내부 함수 */
static NVGpuContext* get_current_gpu(void)
{
    if (!g_state.initialized || g_state.current_device < 0 ||
        g_state.current_device >= g_state.gpu_count) {
        return NULL;
    }
    return g_state.gpu_contexts[g_state.current_device];
}

static void set_error(cudaError_t error)
{
    g_state.last_error = error;
}

/* 전방 선언 */
static void track_host_alloc(void *ptr, bool is_managed);
static bool free_tracked_host_alloc(void *ptr);

/* 버전 정보 */
#define CUDABRIDGE_VERSION_MAJOR 1
#define CUDABRIDGE_VERSION_MINOR 0
#define CUDA_VERSION 12000  /* CUDA 12.0 호환 */

/**
 * CudaBridge 초기화
 */
cudaError_t cudaBridgeInit(void)
{
    if (g_state.initialized) {
        return cudaSuccess;
    }

    CUDA_LOG("Initializing CudaBridge v%d.%d...",
             CUDABRIDGE_VERSION_MAJOR, CUDABRIDGE_VERSION_MINOR);

    pthread_mutex_init(&g_state.lock, NULL);

    /* USB4 컨트롤러 초기화 */
    int ret = usb4_controller_init(&g_state.usb4_ctx);
    if (ret != 0) {
        CUDA_ERR_LOG("Failed to initialize USB4 controller");
        set_error(cudaErrorInitializationError);
        return cudaErrorInitializationError;
    }

    /* USB4 장치 스캔 */
    usb4_scan_routers(&g_state.usb4_ctx);

    /* PCIe 터널 초기화 */
    ret = pcie_tunnel_init(&g_state.pcie_ctx, &g_state.usb4_ctx);
    if (ret != 0) {
        CUDA_ERR_LOG("Failed to initialize PCIe tunnel");
        usb4_controller_shutdown(&g_state.usb4_ctx);
        set_error(cudaErrorInitializationError);
        return cudaErrorInitializationError;
    }

    /* eGPU가 연결되어 있으면 PCIe 터널 생성 */
    for (int i = 0; i < g_state.usb4_ctx.router_count; i++) {
        USB4Router *router = g_state.usb4_ctx.connected_routers[i];
        if (router) {
            USB4PCIeTunnel *tunnel;
            ret = usb4_create_pcie_tunnel(&g_state.usb4_ctx, router, &tunnel);
            if (ret == 0) {
                CUDA_LOG("PCIe tunnel created to USB4 device");
            }
        }
    }

    /* PCIe 버스 스캔 */
    pcie_scan_bus(&g_state.pcie_ctx);

    /* NVIDIA GPU 찾기 */
    g_state.gpu_count = 0;
    for (int i = 0; i < g_state.pcie_ctx.device_count && g_state.gpu_count < 8; i++) {
        PCIeDevice *pcie_dev = g_state.pcie_ctx.devices[i];
        if (pcie_dev && pcie_dev->vendor_id == 0x10DE) {
            NVGpuContext *gpu = calloc(1, sizeof(NVGpuContext));
            if (gpu) {
                ret = nv_gpu_init(gpu, pcie_dev);
                if (ret == 0) {
                    g_state.gpu_contexts[g_state.gpu_count++] = gpu;
                    CUDA_LOG("Found NVIDIA GPU: %s", gpu->info.name);
                } else {
                    free(gpu);
                }
            }
        }
    }

    if (g_state.gpu_count == 0) {
        CUDA_LOG("No NVIDIA GPUs found via eGPU");
        /* 시뮬레이션 모드: 가상 GPU 생성 */
        NVGpuContext *gpu = calloc(1, sizeof(NVGpuContext));
        if (gpu) {
            /* 시뮬레이션용 가상 GPU 설정 */
            gpu->state = NV_GPU_STATE_READY;
            strcpy(gpu->info.name, "CudaBridge Virtual GPU");
            gpu->info.architecture = NV_ARCH_AMPERE;
            gpu->info.vram_size = 8ULL * 1024 * 1024 * 1024;
            gpu->info.compute_cap_major = 8;
            gpu->info.compute_cap_minor = 6;
            gpu->info.sm_info.count = 40;
            gpu->info.sm_info.cores_per_sm = 128;
            gpu->info.max_threads_per_block = 1024;
            gpu->vram_free = gpu->info.vram_size;

            g_state.gpu_contexts[g_state.gpu_count++] = gpu;
            CUDA_LOG("Created virtual GPU for testing");
        }
    }

    g_state.current_device = 0;
    g_state.initialized = true;
    g_state.last_error = cudaSuccess;

    CUDA_LOG("CudaBridge initialized: %d GPU(s) available", g_state.gpu_count);

    return cudaSuccess;
}

/**
 * CudaBridge 종료
 */
cudaError_t cudaBridgeShutdown(void)
{
    if (!g_state.initialized) {
        return cudaSuccess;
    }

    CUDA_LOG("Shutting down CudaBridge...");

    pthread_mutex_lock(&g_state.lock);

    /* GPU 컨텍스트 정리 */
    for (int i = 0; i < g_state.gpu_count; i++) {
        if (g_state.gpu_contexts[i]) {
            nv_gpu_shutdown(g_state.gpu_contexts[i]);
            free(g_state.gpu_contexts[i]);
        }
    }

    /* PCIe 터널 정리 */
    pcie_tunnel_shutdown(&g_state.pcie_ctx);

    /* USB4 컨트롤러 정리 */
    usb4_controller_shutdown(&g_state.usb4_ctx);

    g_state.initialized = false;
    g_state.gpu_count = 0;

    pthread_mutex_unlock(&g_state.lock);
    pthread_mutex_destroy(&g_state.lock);

    CUDA_LOG("CudaBridge shutdown complete");

    return cudaSuccess;
}

/**
 * 드라이버 버전
 */
cudaError_t cudaDriverGetVersion(int *driverVersion)
{
    if (!driverVersion) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    *driverVersion = CUDA_VERSION;
    return cudaSuccess;
}

/**
 * 런타임 버전
 */
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
    if (!runtimeVersion) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }
    *runtimeVersion = CUDA_VERSION;
    return cudaSuccess;
}

/**
 * 디바이스 수 조회
 */
cudaError_t cudaGetDeviceCount(int *count)
{
    if (!count) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    /* 자동 초기화 */
    if (!g_state.initialized) {
        cudaError_t err = cudaBridgeInit();
        if (err != cudaSuccess) {
            *count = 0;
            return err;
        }
    }

    *count = g_state.gpu_count;
    return cudaSuccess;
}

/**
 * 현재 디바이스 조회
 */
cudaError_t cudaGetDevice(int *device)
{
    if (!device) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    if (!g_state.initialized) {
        cudaBridgeInit();
    }

    *device = g_state.current_device;
    return cudaSuccess;
}

/**
 * 디바이스 선택
 */
cudaError_t cudaSetDevice(int device)
{
    if (!g_state.initialized) {
        cudaBridgeInit();
    }

    if (device < 0 || device >= g_state.gpu_count) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    g_state.current_device = device;
    CUDA_DBG("Set current device to %d", device);

    return cudaSuccess;
}

/**
 * 디바이스 속성 조회
 */
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    if (!prop) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    if (!g_state.initialized) {
        cudaBridgeInit();
    }

    if (device < 0 || device >= g_state.gpu_count) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    NVGpuContext *gpu = g_state.gpu_contexts[device];
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    memset(prop, 0, sizeof(cudaDeviceProp));

    /* GPU 정보 복사 */
    strncpy(prop->name, gpu->info.name, sizeof(prop->name) - 1);
    prop->totalGlobalMem = gpu->info.vram_size;
    prop->sharedMemPerBlock = gpu->info.sm_info.shared_mem_per_sm / 2;
    prop->regsPerBlock = gpu->info.sm_info.registers_per_sm;
    prop->warpSize = 32;
    prop->maxThreadsPerBlock = gpu->info.max_threads_per_block;

    prop->maxThreadsDim[0] = gpu->info.max_block_dim[0];
    prop->maxThreadsDim[1] = gpu->info.max_block_dim[1];
    prop->maxThreadsDim[2] = gpu->info.max_block_dim[2];

    prop->maxGridSize[0] = gpu->info.max_grid_dim[0];
    prop->maxGridSize[1] = gpu->info.max_grid_dim[1];
    prop->maxGridSize[2] = gpu->info.max_grid_dim[2];

    prop->clockRate = gpu->info.gpu_clock_mhz * 1000;  /* kHz */
    prop->major = gpu->info.compute_cap_major;
    prop->minor = gpu->info.compute_cap_minor;
    prop->multiProcessorCount = gpu->info.sm_info.count;
    prop->l2CacheSize = 48 * 1024 * 1024;  /* 48 MB (typical) */
    prop->memoryClockRate = gpu->info.mem_clock_mhz * 1000;
    prop->memoryBusWidth = gpu->info.vram_bus_width;
    prop->maxThreadsPerMultiProcessor = gpu->info.sm_info.max_threads_per_sm;
    prop->sharedMemPerMultiprocessor = gpu->info.sm_info.shared_mem_per_sm;
    prop->regsPerMultiprocessor = gpu->info.sm_info.registers_per_sm;

    prop->concurrentKernels = 1;
    prop->asyncEngineCount = 2;
    prop->unifiedAddressing = 1;
    prop->managedMemory = 1;
    prop->cooperativeLaunch = 1;

    return cudaSuccess;
}

/**
 * 디바이스 리셋
 */
cudaError_t cudaDeviceReset(void)
{
    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    int ret = nv_gpu_reset(gpu);
    if (ret != 0) {
        set_error(cudaErrorUnknown);
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

/**
 * 디바이스 동기화
 */
cudaError_t cudaDeviceSynchronize(void)
{
    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    int ret = nv_gpu_synchronize(gpu);
    if (ret != 0) {
        set_error(cudaErrorUnknown);
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

/**
 * 디바이스 속성 조회 (개별)
 */
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device)
{
    if (!value) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return err;
    }

    /* 속성 매핑 (일부) */
    switch (attr) {
        case 1:  /* maxThreadsPerBlock */
            *value = prop.maxThreadsPerBlock;
            break;
        case 2:  /* maxBlockDimX */
            *value = prop.maxThreadsDim[0];
            break;
        case 3:  /* maxBlockDimY */
            *value = prop.maxThreadsDim[1];
            break;
        case 4:  /* maxBlockDimZ */
            *value = prop.maxThreadsDim[2];
            break;
        case 5:  /* maxGridDimX */
            *value = prop.maxGridSize[0];
            break;
        case 10: /* warpSize */
            *value = prop.warpSize;
            break;
        case 16: /* multiProcessorCount */
            *value = prop.multiProcessorCount;
            break;
        case 75: /* computeCapabilityMajor */
            *value = prop.major;
            break;
        case 76: /* computeCapabilityMinor */
            *value = prop.minor;
            break;
        default:
            *value = 0;
    }

    return cudaSuccess;
}

/**
 * 메모리 할당
 */
cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    if (!devPtr || size == 0) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    NVMemoryAlloc alloc = {
        .size = size,
        .type = NV_MEM_TYPE_VIDEO
    };

    int ret = nv_gpu_alloc_memory(gpu, &alloc);
    if (ret != 0) {
        set_error(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }

    /* GPU 주소를 포인터로 반환 (실제 매핑 필요) */
    *devPtr = (void*)(uintptr_t)alloc.gpu_addr;

    CUDA_DBG("cudaMalloc: %zu bytes at %p", size, *devPtr);

    return cudaSuccess;
}

/**
 * 메모리 해제
 */
cudaError_t cudaFree(void *devPtr)
{
    if (!devPtr) {
        return cudaSuccess;  /* NULL은 무시 */
    }

    /* 먼저 호스트/관리 메모리인지 확인 */
    if (free_tracked_host_alloc(devPtr)) {
        CUDA_DBG("cudaFree: %p (managed/host memory)", devPtr);
        return cudaSuccess;
    }

    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    NVMemoryAlloc alloc = {
        .gpu_addr = (uint64_t)(uintptr_t)devPtr
    };

    nv_gpu_free_memory(gpu, &alloc);

    CUDA_DBG("cudaFree: %p", devPtr);

    return cudaSuccess;
}

/**
 * 메모리 복사
 */
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind)
{
    if (!dst || !src || count == 0) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    NVGpuContext *gpu = get_current_gpu();
    if (!gpu && kind != cudaMemcpyHostToHost) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    int ret = 0;

    switch (kind) {
        case cudaMemcpyHostToHost:
            memcpy(dst, src, count);
            break;

        case cudaMemcpyHostToDevice:
            ret = nv_gpu_memcpy_h2d(gpu, (uint64_t)(uintptr_t)dst, src, count);
            break;

        case cudaMemcpyDeviceToHost:
            ret = nv_gpu_memcpy_d2h(gpu, dst, (uint64_t)(uintptr_t)src, count);
            break;

        case cudaMemcpyDeviceToDevice:
            ret = nv_gpu_memcpy_d2d(gpu, (uint64_t)(uintptr_t)dst,
                                    (uint64_t)(uintptr_t)src, count);
            break;

        default:
            set_error(cudaErrorInvalidMemcpyDirection);
            return cudaErrorInvalidMemcpyDirection;
    }

    if (ret != 0) {
        set_error(cudaErrorUnknown);
        return cudaErrorUnknown;
    }

    CUDA_DBG("cudaMemcpy: %zu bytes (%d)", count, kind);

    return cudaSuccess;
}

/**
 * 비동기 메모리 복사
 */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream)
{
    /* 현재는 동기 복사로 구현 */
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

/**
 * 메모리 설정
 */
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    if (!devPtr || count == 0) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    /* 임시 버퍼로 memset 후 복사 */
    void *tmp = malloc(count);
    if (!tmp) {
        set_error(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }

    memset(tmp, value, count);
    cudaError_t err = cudaMemcpy(devPtr, tmp, count, cudaMemcpyHostToDevice);
    free(tmp);

    return err;
}

/**
 * 비동기 메모리 설정
 */
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream)
{
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

/* 호스트 메모리 추적 추가 */
static void track_host_alloc(void *ptr, bool is_managed)
{
    if (g_state.host_alloc_count < MAX_HOST_ALLOCS) {
        g_state.host_allocs[g_state.host_alloc_count].ptr = ptr;
        g_state.host_allocs[g_state.host_alloc_count].is_managed = is_managed;
        g_state.host_alloc_count++;
    }
}

/* 호스트 메모리 추적에서 제거하고 해제 */
static bool free_tracked_host_alloc(void *ptr)
{
    for (int i = 0; i < g_state.host_alloc_count; i++) {
        if (g_state.host_allocs[i].ptr == ptr) {
            free(ptr);
            /* 배열에서 제거 */
            for (int j = i; j < g_state.host_alloc_count - 1; j++) {
                g_state.host_allocs[j] = g_state.host_allocs[j + 1];
            }
            g_state.host_alloc_count--;
            return true;
        }
    }
    return false;
}

/**
 * 호스트 메모리 할당 (페이지 락)
 */
cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    if (!ptr || size == 0) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    /* posix_memalign으로 정렬된 메모리 할당 */
    int ret = posix_memalign(ptr, 4096, size);
    if (ret != 0) {
        set_error(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }

    /* 추적 추가 */
    track_host_alloc(*ptr, false);

    return cudaSuccess;
}

/**
 * 호스트 메모리 해제
 */
cudaError_t cudaFreeHost(void *ptr)
{
    if (ptr) {
        free_tracked_host_alloc(ptr);
    }
    return cudaSuccess;
}

/**
 * 호스트 메모리 할당 (플래그)
 */
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    (void)flags;
    return cudaMallocHost(pHost, size);
}

/**
 * 호스트 포인터에서 디바이스 포인터 획득
 */
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                     unsigned int flags)
{
    (void)flags;
    if (!pDevice || !pHost) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    /* 통합 주소 지정: 동일 주소 반환 */
    *pDevice = pHost;
    return cudaSuccess;
}

/**
 * 메모리 정보
 */
cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
    if (!free || !total) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    *total = gpu->info.vram_size;
    *free = gpu->vram_free;

    return cudaSuccess;
}

/**
 * 스트림 생성
 */
cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    return cudaStreamCreateWithFlags(pStream, 0);
}

/**
 * 스트림 생성 (플래그)
 */
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    (void)flags;

    if (!pStream) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    struct CUstream_st *stream = calloc(1, sizeof(struct CUstream_st));
    if (!stream) {
        set_error(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }

    stream->device = g_state.current_device;
    stream->is_default = false;

    /* GPU 채널 생성 */
    int ret = nv_gpu_create_channel(gpu, &stream->channel);
    if (ret != 0) {
        free(stream);
        set_error(cudaErrorUnknown);
        return cudaErrorUnknown;
    }

    *pStream = stream;

    CUDA_DBG("Created stream %p", stream);

    return cudaSuccess;
}

/**
 * 스트림 제거
 */
cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    if (!stream) {
        return cudaSuccess;
    }

    NVGpuContext *gpu = g_state.gpu_contexts[stream->device];
    if (gpu && stream->channel) {
        nv_gpu_destroy_channel(gpu, stream->channel);
    }

    CUDA_DBG("Destroyed stream %p", stream);

    free(stream);
    return cudaSuccess;
}

/**
 * 스트림 동기화
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    if (!stream) {
        return cudaDeviceSynchronize();
    }

    NVGpuContext *gpu = g_state.gpu_contexts[stream->device];
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    if (stream->channel) {
        nv_gpu_channel_sync(gpu, stream->channel);
    }

    return cudaSuccess;
}

/**
 * 스트림 쿼리
 */
cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    (void)stream;
    /* 모든 작업이 완료되었다고 가정 */
    return cudaSuccess;
}

/**
 * 스트림이 이벤트 대기
 */
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags)
{
    (void)stream;
    (void)event;
    (void)flags;
    return cudaSuccess;
}

/**
 * 이벤트 생성
 */
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event, 0);
}

/**
 * 이벤트 생성 (플래그)
 */
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    (void)flags;

    if (!event) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    struct CUevent_st *ev = calloc(1, sizeof(struct CUevent_st));
    if (!ev) {
        set_error(cudaErrorMemoryAllocation);
        return cudaErrorMemoryAllocation;
    }

    ev->device = g_state.current_device;
    ev->recorded = false;

    *event = ev;
    return cudaSuccess;
}

/**
 * 이벤트 제거
 */
cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    free(event);
    return cudaSuccess;
}

/**
 * 이벤트 기록
 */
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    (void)stream;

    if (!event) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    /* 현재 시간 기록 */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    event->timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    event->recorded = true;

    return cudaSuccess;
}

/**
 * 이벤트 동기화
 */
cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    (void)event;
    return cudaSuccess;
}

/**
 * 이벤트 쿼리
 */
cudaError_t cudaEventQuery(cudaEvent_t event)
{
    (void)event;
    return cudaSuccess;
}

/**
 * 이벤트 경과 시간
 */
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    if (!ms || !start || !end) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    if (!start->recorded || !end->recorded) {
        set_error(cudaErrorInvalidResourceHandle);
        return cudaErrorInvalidResourceHandle;
    }

    uint64_t diff = end->timestamp - start->timestamp;
    *ms = (float)diff / 1000000.0f;  /* ns to ms */

    return cudaSuccess;
}

/**
 * 마지막 에러 가져오기
 */
cudaError_t cudaGetLastError(void)
{
    cudaError_t err = g_state.last_error;
    g_state.last_error = cudaSuccess;
    return err;
}

/**
 * 마지막 에러 확인 (리셋 없이)
 */
cudaError_t cudaPeekAtLastError(void)
{
    return g_state.last_error;
}

/**
 * 에러 이름
 */
const char* cudaGetErrorName(cudaError_t error)
{
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorNoDevice: return "cudaErrorNoDevice";
        case cudaErrorNotReady: return "cudaErrorNotReady";
        default: return "cudaErrorUnknown";
    }
}

/**
 * 에러 설명
 */
const char* cudaGetErrorString(cudaError_t error)
{
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidDevice:
            return "invalid device ordinal";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid memcpy direction";
        case cudaErrorNoDevice:
            return "no CUDA-capable device is detected";
        case cudaErrorNotReady:
            return "device not ready";
        default:
            return "unknown error";
    }
}

/**
 * 커널 실행
 */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream)
{
    NVGpuContext *gpu = get_current_gpu();
    if (!gpu) {
        set_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    NVChannel *channel = NULL;
    if (stream && stream->channel) {
        channel = stream->channel;
    } else if (gpu->channel_count > 0) {
        channel = gpu->channels[0];
    }

    if (!channel) {
        /* 기본 채널 생성 */
        int ret = nv_gpu_create_channel(gpu, &channel);
        if (ret != 0) {
            set_error(cudaErrorLaunchFailure);
            return cudaErrorLaunchFailure;
        }
    }

    NVKernelParams params = {
        .func_addr = (uint64_t)(uintptr_t)func,
        .grid_dim = { gridDim.x, gridDim.y, gridDim.z },
        .block_dim = { blockDim.x, blockDim.y, blockDim.z },
        .shared_mem = (uint32_t)sharedMem,
        .channel = channel,
        .args = args,
        .arg_sizes = NULL,
        .arg_count = 0  /* 실제로는 args 파싱 필요 */
    };

    int ret = nv_gpu_launch_kernel(gpu, &params);
    if (ret != 0) {
        set_error(cudaErrorLaunchFailure);
        return cudaErrorLaunchFailure;
    }

    CUDA_DBG("Launched kernel at %p: grid=(%d,%d,%d) block=(%d,%d,%d)",
             func, gridDim.x, gridDim.y, gridDim.z,
             blockDim.x, blockDim.y, blockDim.z);

    return cudaSuccess;
}

/**
 * 연결 상태 확인
 */
cudaError_t cudaBridgeGetConnectionStatus(int *isConnected)
{
    if (!isConnected) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    if (!g_state.initialized) {
        *isConnected = 0;
        return cudaSuccess;
    }

    /* USB4 터널 상태 확인 */
    *isConnected = (g_state.usb4_ctx.tunnel_count > 0) ? 1 : 0;

    return cudaSuccess;
}

/**
 * 대역폭 정보
 */
cudaError_t cudaBridgeGetBandwidthInfo(size_t *upstreamBw, size_t *downstreamBw)
{
    if (!upstreamBw || !downstreamBw) {
        set_error(cudaErrorInvalidValue);
        return cudaErrorInvalidValue;
    }

    if (!g_state.initialized || g_state.usb4_ctx.tunnel_count == 0) {
        *upstreamBw = 0;
        *downstreamBw = 0;
        return cudaSuccess;
    }

    /* USB4 Gen 3 기준 양방향 각 32 Gbps */
    *upstreamBw = 32ULL * 1024 * 1024 * 1024 / 8;    /* bytes/sec */
    *downstreamBw = 32ULL * 1024 * 1024 * 1024 / 8;

    return cudaSuccess;
}
