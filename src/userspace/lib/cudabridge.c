/**
 * CudaBridge - Userspace Library Implementation
 *
 * 공개 API 구현
 */

#include "../include/cudabridge.h"
#include "../../cuda/runtime/cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 내부 상태 */
static int g_initialized = 0;
static cbError_t g_last_error = cbSuccess;
static void (*g_hotplug_callback)(int, int) = NULL;

/* 에러 설정 매크로 */
#define SET_ERROR(err) do { g_last_error = (err); } while(0)

/* CUDA 에러를 CB 에러로 변환 */
static cbError_t cuda_to_cb_error(cudaError_t cuda_err)
{
    switch (cuda_err) {
        case cudaSuccess: return cbSuccess;
        case cudaErrorInvalidValue: return cbErrorInvalidValue;
        case cudaErrorMemoryAllocation: return cbErrorOutOfMemory;
        case cudaErrorInitializationError: return cbErrorNotInitialized;
        case cudaErrorNoDevice: return cbErrorNoDevice;
        case cudaErrorInvalidDevice: return cbErrorInvalidDevice;
        default: return cbErrorUnknown;
    }
}

/* ========== 초기화 및 종료 ========== */

cbError_t cbInit(void)
{
    if (g_initialized) {
        return cbSuccess;
    }

    cudaError_t err = cudaBridgeInit();
    if (err != cudaSuccess) {
        SET_ERROR(cuda_to_cb_error(err));
        return g_last_error;
    }

    g_initialized = 1;
    return cbSuccess;
}

cbError_t cbShutdown(void)
{
    if (!g_initialized) {
        return cbSuccess;
    }

    cudaBridgeShutdown();
    g_initialized = 0;
    g_last_error = cbSuccess;
    g_hotplug_callback = NULL;  /* 보안: 댕글링 콜백 포인터 방지 */

    return cbSuccess;
}

int cbIsInitialized(void)
{
    return g_initialized;
}

/* ========== 버전 정보 ========== */

cbError_t cbGetVersion(int *version)
{
    if (!version) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    *version = CUDABRIDGE_VERSION;
    return cbSuccess;
}

cbError_t cbGetDriverVersion(int *version)
{
    if (!version) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaDriverGetVersion(version);
    return cuda_to_cb_error(err);
}

/* ========== 디바이스 관리 ========== */

cbError_t cbGetDeviceCount(int *count)
{
    if (!count) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    /* 자동 초기화 */
    if (!g_initialized) {
        cbError_t err = cbInit();
        if (err != cbSuccess) {
            *count = 0;
            return err;
        }
    }

    cudaError_t err = cudaGetDeviceCount(count);
    return cuda_to_cb_error(err);
}

cbError_t cbGetDevice(int *device)
{
    if (!device) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaGetDevice(device);
    return cuda_to_cb_error(err);
}

cbError_t cbSetDevice(int device)
{
    cudaError_t err = cudaSetDevice(device);
    cbError_t cb_err = cuda_to_cb_error(err);
    if (cb_err != cbSuccess) {
        SET_ERROR(cb_err);
    }
    return cb_err;
}

cbError_t cbGetDeviceProperties(cbDeviceProp *prop, int device)
{
    if (!prop) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaDeviceProp cuda_prop;
    cudaError_t err = cudaGetDeviceProperties(&cuda_prop, device);
    if (err != cudaSuccess) {
        return cuda_to_cb_error(err);
    }

    /* 구조체 복사 */
    memset(prop, 0, sizeof(cbDeviceProp));
    strncpy(prop->name, cuda_prop.name, sizeof(prop->name) - 1);
    prop->totalGlobalMem = cuda_prop.totalGlobalMem;
    prop->sharedMemPerBlock = cuda_prop.sharedMemPerBlock;
    prop->regsPerBlock = cuda_prop.regsPerBlock;
    prop->warpSize = cuda_prop.warpSize;
    prop->maxThreadsPerBlock = cuda_prop.maxThreadsPerBlock;
    memcpy(prop->maxThreadsDim, cuda_prop.maxThreadsDim, sizeof(prop->maxThreadsDim));
    memcpy(prop->maxGridSize, cuda_prop.maxGridSize, sizeof(prop->maxGridSize));
    prop->clockRate = cuda_prop.clockRate;
    prop->major = cuda_prop.major;
    prop->minor = cuda_prop.minor;
    prop->multiProcessorCount = cuda_prop.multiProcessorCount;
    prop->memoryClockRate = cuda_prop.memoryClockRate;
    prop->memoryBusWidth = cuda_prop.memoryBusWidth;
    prop->l2CacheSize = cuda_prop.l2CacheSize;
    prop->maxThreadsPerMultiProcessor = cuda_prop.maxThreadsPerMultiProcessor;
    prop->concurrentKernels = cuda_prop.concurrentKernels;
    prop->asyncEngineCount = cuda_prop.asyncEngineCount;
    prop->unifiedAddressing = cuda_prop.unifiedAddressing;
    prop->managedMemory = cuda_prop.managedMemory;

    return cbSuccess;
}

cbError_t cbDeviceSynchronize(void)
{
    cudaError_t err = cudaDeviceSynchronize();
    return cuda_to_cb_error(err);
}

cbError_t cbDeviceReset(void)
{
    cudaError_t err = cudaDeviceReset();
    return cuda_to_cb_error(err);
}

/* ========== 메모리 관리 ========== */

cbError_t cbMalloc(void **devPtr, size_t size)
{
    if (!devPtr || size == 0) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaMalloc(devPtr, size);
    cbError_t cb_err = cuda_to_cb_error(err);
    if (cb_err != cbSuccess) {
        SET_ERROR(cb_err);
    }
    return cb_err;
}

cbError_t cbFree(void *devPtr)
{
    cudaError_t err = cudaFree(devPtr);
    return cuda_to_cb_error(err);
}

cbError_t cbMemcpy(void *dst, const void *src, size_t count, cbMemcpyKind kind)
{
    if (!dst || !src || count == 0) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case CB_MEMCPY_HOST_TO_HOST:     cuda_kind = cudaMemcpyHostToHost; break;
        case CB_MEMCPY_HOST_TO_DEVICE:   cuda_kind = cudaMemcpyHostToDevice; break;
        case CB_MEMCPY_DEVICE_TO_HOST:   cuda_kind = cudaMemcpyDeviceToHost; break;
        case CB_MEMCPY_DEVICE_TO_DEVICE: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default:                         cuda_kind = cudaMemcpyDefault; break;
    }

    cudaError_t err = cudaMemcpy(dst, src, count, cuda_kind);
    cbError_t cb_err = cuda_to_cb_error(err);
    if (cb_err != cbSuccess) {
        SET_ERROR(cb_err);
    }
    return cb_err;
}

cbError_t cbMemcpyAsync(void *dst, const void *src, size_t count,
                        cbMemcpyKind kind, cbStream_t stream)
{
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case CB_MEMCPY_HOST_TO_HOST:     cuda_kind = cudaMemcpyHostToHost; break;
        case CB_MEMCPY_HOST_TO_DEVICE:   cuda_kind = cudaMemcpyHostToDevice; break;
        case CB_MEMCPY_DEVICE_TO_HOST:   cuda_kind = cudaMemcpyDeviceToHost; break;
        case CB_MEMCPY_DEVICE_TO_DEVICE: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default:                         cuda_kind = cudaMemcpyDefault; break;
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, count, cuda_kind,
                                       (cudaStream_t)stream);
    return cuda_to_cb_error(err);
}

cbError_t cbMemset(void *devPtr, int value, size_t count)
{
    cudaError_t err = cudaMemset(devPtr, value, count);
    return cuda_to_cb_error(err);
}

cbError_t cbMallocHost(void **ptr, size_t size)
{
    if (!ptr || size == 0) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaMallocHost(ptr, size);
    return cuda_to_cb_error(err);
}

cbError_t cbFreeHost(void *ptr)
{
    cudaError_t err = cudaFreeHost(ptr);
    return cuda_to_cb_error(err);
}

cbError_t cbMallocManaged(void **devPtr, size_t size)
{
    if (!devPtr || size == 0) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    /* 통합 메모리는 현재 호스트 할당으로 시뮬레이션 */
    cudaError_t err = cudaMallocHost(devPtr, size);
    return cuda_to_cb_error(err);
}

cbError_t cbMemGetInfo(size_t *free, size_t *total)
{
    cudaError_t err = cudaMemGetInfo(free, total);
    return cuda_to_cb_error(err);
}

/* ========== 스트림 관리 ========== */

cbError_t cbStreamCreate(cbStream_t *pStream)
{
    if (!pStream) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaStreamCreate((cudaStream_t*)pStream);
    return cuda_to_cb_error(err);
}

cbError_t cbStreamDestroy(cbStream_t stream)
{
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    return cuda_to_cb_error(err);
}

cbError_t cbStreamSynchronize(cbStream_t stream)
{
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    return cuda_to_cb_error(err);
}

cbError_t cbStreamQuery(cbStream_t stream)
{
    cudaError_t err = cudaStreamQuery((cudaStream_t)stream);
    return cuda_to_cb_error(err);
}

/* ========== 이벤트 관리 ========== */

cbError_t cbEventCreate(cbEvent_t *event)
{
    if (!event) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaEventCreate((cudaEvent_t*)event);
    return cuda_to_cb_error(err);
}

cbError_t cbEventDestroy(cbEvent_t event)
{
    cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
    return cuda_to_cb_error(err);
}

cbError_t cbEventRecord(cbEvent_t event, cbStream_t stream)
{
    cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    return cuda_to_cb_error(err);
}

cbError_t cbEventSynchronize(cbEvent_t event)
{
    cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
    return cuda_to_cb_error(err);
}

cbError_t cbEventElapsedTime(float *ms, cbEvent_t start, cbEvent_t end)
{
    cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start,
                                           (cudaEvent_t)end);
    return cuda_to_cb_error(err);
}

/* ========== 커널 실행 ========== */

cbError_t cbLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                         void **args, size_t sharedMem, cbStream_t stream)
{
    cudaError_t err = cudaLaunchKernel(func, gridDim, blockDim, args,
                                       sharedMem, (cudaStream_t)stream);
    cbError_t cb_err = cuda_to_cb_error(err);
    if (cb_err != cbSuccess) {
        SET_ERROR(cb_err);
    }
    return cb_err;
}

/* ========== 에러 처리 ========== */

cbError_t cbGetLastError(void)
{
    cbError_t err = g_last_error;
    g_last_error = cbSuccess;
    return err;
}

cbError_t cbPeekAtLastError(void)
{
    return g_last_error;
}

const char* cbGetErrorName(cbError_t error)
{
    switch (error) {
        case cbSuccess: return "cbSuccess";
        case cbErrorInvalidValue: return "cbErrorInvalidValue";
        case cbErrorOutOfMemory: return "cbErrorOutOfMemory";
        case cbErrorNotInitialized: return "cbErrorNotInitialized";
        case cbErrorNoDevice: return "cbErrorNoDevice";
        case cbErrorInvalidDevice: return "cbErrorInvalidDevice";
        case cbErrorDeviceNotConnected: return "cbErrorDeviceNotConnected";
        case cbErrorDriverNotFound: return "cbErrorDriverNotFound";
        case cbErrorTimeout: return "cbErrorTimeout";
        case cbErrorLaunchFailure: return "cbErrorLaunchFailure";
        case cbErrorInvalidKernel: return "cbErrorInvalidKernel";
        case cbErrorInvalidConfiguration: return "cbErrorInvalidConfiguration";
        case cbErrorEGPUNotSupported: return "cbErrorEGPUNotSupported";
        case cbErrorUSB4NotAvailable: return "cbErrorUSB4NotAvailable";
        case cbErrorBandwidthInsufficient: return "cbErrorBandwidthInsufficient";
        default: return "cbErrorUnknown";
    }
}

const char* cbGetErrorString(cbError_t error)
{
    switch (error) {
        case cbSuccess:
            return "no error";
        case cbErrorInvalidValue:
            return "invalid argument";
        case cbErrorOutOfMemory:
            return "out of memory";
        case cbErrorNotInitialized:
            return "CudaBridge not initialized";
        case cbErrorNoDevice:
            return "no CUDA-capable device detected";
        case cbErrorInvalidDevice:
            return "invalid device ordinal";
        case cbErrorDeviceNotConnected:
            return "eGPU device not connected";
        case cbErrorDriverNotFound:
            return "driver not found";
        case cbErrorTimeout:
            return "operation timed out";
        case cbErrorLaunchFailure:
            return "kernel launch failure";
        case cbErrorInvalidKernel:
            return "invalid kernel";
        case cbErrorInvalidConfiguration:
            return "invalid configuration";
        case cbErrorEGPUNotSupported:
            return "eGPU not supported on this system";
        case cbErrorUSB4NotAvailable:
            return "USB4/Thunderbolt not available";
        case cbErrorBandwidthInsufficient:
            return "insufficient bandwidth";
        default:
            return "unknown error";
    }
}

/* ========== eGPU/USB4 전용 기능 ========== */

cbError_t cbGetConnectionInfo(cbConnectionInfo *info)
{
    if (!info) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    memset(info, 0, sizeof(cbConnectionInfo));

    int connected;
    cudaBridgeGetConnectionStatus(&connected);

    info->isConnected = connected;
    info->connectionType = 0;  /* USB4 */

    if (connected) {
        size_t upBw, downBw;
        cudaBridgeGetBandwidthInfo(&upBw, &downBw);
        info->upstreamBandwidth = upBw;
        info->downstreamBandwidth = downBw;
        info->pcieGeneration = 3;  /* USB4는 PCIe 3.0 수준 */
        info->pcieLanes = 4;
        info->linkUtilization = 0.0f;
    }

    return cbSuccess;
}

cbError_t cbGetUSB4Bandwidth(size_t *upBw, size_t *downBw)
{
    if (!upBw || !downBw) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    cudaError_t err = cudaBridgeGetBandwidthInfo(upBw, downBw);
    return cuda_to_cb_error(err);
}

cbError_t cbSetHotplugCallback(void (*callback)(int device, int connected))
{
    g_hotplug_callback = callback;
    return cbSuccess;
}

cbError_t cbGetPCIeTunnelStatus(int *isActive, int *bandwidth)
{
    if (!isActive || !bandwidth) {
        SET_ERROR(cbErrorInvalidValue);
        return cbErrorInvalidValue;
    }

    int connected;
    cudaBridgeGetConnectionStatus(&connected);

    *isActive = connected;
    *bandwidth = connected ? 32000 : 0;  /* 32 Gbps */

    return cbSuccess;
}
