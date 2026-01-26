/**
 * CudaBridge - CUDA Runtime API Compatibility Layer
 *
 * NVIDIA CUDA Runtime API와 호환되는 인터페이스를 제공합니다.
 * 기존 CUDA 애플리케이션을 최소한의 수정으로 사용할 수 있습니다.
 */

#ifndef CUDABRIDGE_CUDA_RUNTIME_H
#define CUDABRIDGE_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA 에러 코드 (NVIDIA CUDA와 호환) */
typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorUnknown = 30,
    cudaErrorInvalidResourceHandle = 33,
    cudaErrorInsufficientDriver = 35,
    cudaErrorNoDevice = 38,
    cudaErrorSetOnActiveProcess = 36,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorLaunchFailure = 719,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorNotReady = 600,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713
} cudaError_t;

/* 메모리 복사 방향 */
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

/* 디바이스 속성 */
typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
} cudaDeviceProp;

/* 스트림 */
typedef struct CUstream_st *cudaStream_t;

/* 이벤트 */
typedef struct CUevent_st *cudaEvent_t;

/* dim3 구조체 */
#ifndef __DIM3_DEFINED__
#define __DIM3_DEFINED__
typedef struct dim3 {
    unsigned int x, y, z;
} dim3;
#endif

/* 초기화 및 버전 */
cudaError_t cudaDriverGetVersion(int *driverVersion);
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);

/* 디바이스 관리 */
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device);

/* 메모리 관리 */
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                     unsigned int flags);
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);

/* 스트림 관리 */
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                unsigned int flags);

/* 이벤트 관리 */
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/* 에러 처리 */
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

/* 커널 실행 (내부용) */
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream);

/* 확장 함수들 (CudaBridge 전용) */

/**
 * CudaBridge 초기화
 * CUDA API 사용 전에 호출 필요
 */
cudaError_t cudaBridgeInit(void);

/**
 * CudaBridge 종료
 * 모든 CUDA 작업 후 호출
 */
cudaError_t cudaBridgeShutdown(void);

/**
 * USB4/eGPU 연결 상태 확인
 */
cudaError_t cudaBridgeGetConnectionStatus(int *isConnected);

/**
 * eGPU 대역폭 정보 쿼리
 */
cudaError_t cudaBridgeGetBandwidthInfo(size_t *upstreamBw,
                                       size_t *downstreamBw);

#ifdef __cplusplus
}
#endif

/* C++ 커널 실행 매크로 */
#ifdef __cplusplus

#define CUDA_KERNEL_LAUNCH(kernel, gridDim, blockDim, sharedMem, stream, ...) \
    do { \
        void *__args[] = { __VA_ARGS__ }; \
        cudaLaunchKernel((const void*)kernel, gridDim, blockDim, \
                         __args, sharedMem, stream); \
    } while(0)

#endif

#endif /* CUDABRIDGE_CUDA_RUNTIME_H */
