/**
 * CudaBridge - Public API Header
 *
 * Apple Silicon Mac에서 eGPU를 통한 CUDA 연산을 위한 공개 API입니다.
 *
 * Usage:
 *   #include <cudabridge.h>
 *
 *   cbInit();
 *   // ... CUDA 작업 ...
 *   cbShutdown();
 *
 * 또는 CUDA 호환 모드:
 *   #define CUDABRIDGE_COMPAT_MODE
 *   #include <cudabridge.h>
 *   // 기존 CUDA 코드 그대로 사용 가능
 */

#ifndef CUDABRIDGE_H
#define CUDABRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 버전 정보 */
#define CUDABRIDGE_VERSION_MAJOR 1
#define CUDABRIDGE_VERSION_MINOR 0
#define CUDABRIDGE_VERSION_PATCH 0
#define CUDABRIDGE_VERSION \
    ((CUDABRIDGE_VERSION_MAJOR * 10000) + \
     (CUDABRIDGE_VERSION_MINOR * 100) + \
     CUDABRIDGE_VERSION_PATCH)

/* 에러 코드 */
typedef enum cbError {
    cbSuccess = 0,
    cbErrorInvalidValue = 1,
    cbErrorOutOfMemory = 2,
    cbErrorNotInitialized = 3,
    cbErrorNoDevice = 4,
    cbErrorInvalidDevice = 5,
    cbErrorDeviceNotConnected = 6,
    cbErrorDriverNotFound = 7,
    cbErrorTimeout = 8,
    cbErrorLaunchFailure = 9,
    cbErrorInvalidKernel = 10,
    cbErrorInvalidConfiguration = 11,
    cbErrorEGPUNotSupported = 12,
    cbErrorUSB4NotAvailable = 13,
    cbErrorBandwidthInsufficient = 14,
    cbErrorUnknown = 999
} cbError_t;

/* 메모리 복사 방향 */
typedef enum cbMemcpyKind {
    CB_MEMCPY_HOST_TO_HOST = 0,
    CB_MEMCPY_HOST_TO_DEVICE = 1,
    CB_MEMCPY_DEVICE_TO_HOST = 2,
    CB_MEMCPY_DEVICE_TO_DEVICE = 3,
    CB_MEMCPY_DEFAULT = 4
} cbMemcpyKind;

/* 디바이스 속성 */
typedef struct cbDeviceProp {
    char        name[256];              /* GPU 이름 */
    size_t      totalGlobalMem;         /* 전체 VRAM */
    size_t      sharedMemPerBlock;      /* 블록당 공유 메모리 */
    int         regsPerBlock;           /* 블록당 레지스터 */
    int         warpSize;               /* 워프 크기 */
    int         maxThreadsPerBlock;     /* 블록당 최대 스레드 */
    int         maxThreadsDim[3];       /* 최대 블록 차원 */
    int         maxGridSize[3];         /* 최대 그리드 차원 */
    int         clockRate;              /* 클럭 (kHz) */
    int         major;                  /* Compute capability major */
    int         minor;                  /* Compute capability minor */
    int         multiProcessorCount;    /* SM 수 */
    int         memoryClockRate;        /* 메모리 클럭 (kHz) */
    int         memoryBusWidth;         /* 메모리 버스 폭 (bits) */
    int         l2CacheSize;            /* L2 캐시 크기 */
    int         maxThreadsPerMultiProcessor; /* SM당 최대 스레드 */
    int         computeMode;            /* 컴퓨트 모드 */
    int         concurrentKernels;      /* 동시 커널 지원 */
    int         asyncEngineCount;       /* 비동기 엔진 수 */
    int         unifiedAddressing;      /* 통합 주소 지정 */
    int         managedMemory;          /* 통합 메모리 지원 */
} cbDeviceProp;

/* 연결 정보 */
typedef struct cbConnectionInfo {
    int         isConnected;            /* 연결 상태 */
    int         connectionType;         /* 0: USB4, 1: TB3, 2: TB4 */
    size_t      upstreamBandwidth;      /* 업스트림 대역폭 (bytes/s) */
    size_t      downstreamBandwidth;    /* 다운스트림 대역폭 (bytes/s) */
    int         pcieGeneration;         /* PCIe 세대 */
    int         pcieLanes;              /* PCIe 레인 수 */
    float       linkUtilization;        /* 링크 사용률 (0.0-1.0) */
} cbConnectionInfo;

/* 스트림 및 이벤트 핸들 */
typedef struct cbStream_st *cbStream_t;
typedef struct cbEvent_st *cbEvent_t;

/* dim3 구조체 (커널 실행용) */
#ifndef __DIM3_DEFINED__
#define __DIM3_DEFINED__
#ifndef __cplusplus
typedef struct dim3 {
    unsigned int x, y, z;
} dim3;
#endif
#endif

/* ========== 초기화 및 종료 ========== */

/**
 * CudaBridge 초기화
 * 모든 다른 API 호출 전에 반드시 호출해야 합니다.
 *
 * @return cbSuccess 성공, 그 외 에러 코드
 */
cbError_t cbInit(void);

/**
 * CudaBridge 종료
 * 모든 리소스를 해제합니다.
 *
 * @return cbSuccess 성공
 */
cbError_t cbShutdown(void);

/**
 * 초기화 상태 확인
 *
 * @return 1 초기화됨, 0 아님
 */
int cbIsInitialized(void);

/* ========== 버전 정보 ========== */

/**
 * CudaBridge 버전 조회
 *
 * @param version 버전 번호 출력 (MAJOR*10000 + MINOR*100 + PATCH)
 * @return cbSuccess 성공
 */
cbError_t cbGetVersion(int *version);

/**
 * 드라이버 버전 조회
 *
 * @param version 버전 번호 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetDriverVersion(int *version);

/* ========== 디바이스 관리 ========== */

/**
 * 사용 가능한 디바이스 수 조회
 *
 * @param count 디바이스 수 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetDeviceCount(int *count);

/**
 * 현재 디바이스 조회
 *
 * @param device 디바이스 번호 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetDevice(int *device);

/**
 * 디바이스 선택
 *
 * @param device 디바이스 번호
 * @return cbSuccess 성공
 */
cbError_t cbSetDevice(int device);

/**
 * 디바이스 속성 조회
 *
 * @param prop 속성 구조체 출력
 * @param device 디바이스 번호
 * @return cbSuccess 성공
 */
cbError_t cbGetDeviceProperties(cbDeviceProp *prop, int device);

/**
 * 디바이스 동기화
 * 현재 디바이스의 모든 작업이 완료될 때까지 대기합니다.
 *
 * @return cbSuccess 성공
 */
cbError_t cbDeviceSynchronize(void);

/**
 * 디바이스 리셋
 *
 * @return cbSuccess 성공
 */
cbError_t cbDeviceReset(void);

/* ========== 메모리 관리 ========== */

/**
 * 디바이스 메모리 할당
 *
 * @param devPtr 할당된 포인터 출력
 * @param size 할당 크기 (bytes)
 * @return cbSuccess 성공
 */
cbError_t cbMalloc(void **devPtr, size_t size);

/**
 * 디바이스 메모리 해제
 *
 * @param devPtr 해제할 포인터
 * @return cbSuccess 성공
 */
cbError_t cbFree(void *devPtr);

/**
 * 메모리 복사
 *
 * @param dst 대상 포인터
 * @param src 소스 포인터
 * @param count 복사 크기 (bytes)
 * @param kind 복사 방향
 * @return cbSuccess 성공
 */
cbError_t cbMemcpy(void *dst, const void *src, size_t count, cbMemcpyKind kind);

/**
 * 비동기 메모리 복사
 *
 * @param dst 대상 포인터
 * @param src 소스 포인터
 * @param count 복사 크기 (bytes)
 * @param kind 복사 방향
 * @param stream 스트림 (NULL이면 기본 스트림)
 * @return cbSuccess 성공
 */
cbError_t cbMemcpyAsync(void *dst, const void *src, size_t count,
                        cbMemcpyKind kind, cbStream_t stream);

/**
 * 메모리 설정
 *
 * @param devPtr 대상 포인터
 * @param value 설정할 값
 * @param count 크기 (bytes)
 * @return cbSuccess 성공
 */
cbError_t cbMemset(void *devPtr, int value, size_t count);

/**
 * 호스트 메모리 할당 (페이지 락)
 *
 * @param ptr 할당된 포인터 출력
 * @param size 할당 크기 (bytes)
 * @return cbSuccess 성공
 */
cbError_t cbMallocHost(void **ptr, size_t size);

/**
 * 호스트 메모리 해제
 *
 * @param ptr 해제할 포인터
 * @return cbSuccess 성공
 */
cbError_t cbFreeHost(void *ptr);

/**
 * 통합 메모리 할당
 *
 * @param devPtr 할당된 포인터 출력
 * @param size 할당 크기 (bytes)
 * @return cbSuccess 성공
 */
cbError_t cbMallocManaged(void **devPtr, size_t size);

/**
 * 메모리 정보 조회
 *
 * @param free 사용 가능한 메모리 출력
 * @param total 전체 메모리 출력
 * @return cbSuccess 성공
 */
cbError_t cbMemGetInfo(size_t *free, size_t *total);

/* ========== 스트림 관리 ========== */

/**
 * 스트림 생성
 *
 * @param pStream 스트림 핸들 출력
 * @return cbSuccess 성공
 */
cbError_t cbStreamCreate(cbStream_t *pStream);

/**
 * 스트림 제거
 *
 * @param stream 스트림 핸들
 * @return cbSuccess 성공
 */
cbError_t cbStreamDestroy(cbStream_t stream);

/**
 * 스트림 동기화
 *
 * @param stream 스트림 핸들
 * @return cbSuccess 성공
 */
cbError_t cbStreamSynchronize(cbStream_t stream);

/**
 * 스트림 쿼리
 *
 * @param stream 스트림 핸들
 * @return cbSuccess 완료됨, cbErrorNotReady 진행 중
 */
cbError_t cbStreamQuery(cbStream_t stream);

/* ========== 이벤트 관리 ========== */

/**
 * 이벤트 생성
 *
 * @param event 이벤트 핸들 출력
 * @return cbSuccess 성공
 */
cbError_t cbEventCreate(cbEvent_t *event);

/**
 * 이벤트 제거
 *
 * @param event 이벤트 핸들
 * @return cbSuccess 성공
 */
cbError_t cbEventDestroy(cbEvent_t event);

/**
 * 이벤트 기록
 *
 * @param event 이벤트 핸들
 * @param stream 스트림 핸들 (NULL이면 기본 스트림)
 * @return cbSuccess 성공
 */
cbError_t cbEventRecord(cbEvent_t event, cbStream_t stream);

/**
 * 이벤트 동기화
 *
 * @param event 이벤트 핸들
 * @return cbSuccess 성공
 */
cbError_t cbEventSynchronize(cbEvent_t event);

/**
 * 이벤트 경과 시간
 *
 * @param ms 경과 시간 (밀리초) 출력
 * @param start 시작 이벤트
 * @param end 종료 이벤트
 * @return cbSuccess 성공
 */
cbError_t cbEventElapsedTime(float *ms, cbEvent_t start, cbEvent_t end);

/* ========== 커널 실행 ========== */

/**
 * 커널 실행
 *
 * @param func 커널 함수 포인터
 * @param gridDim 그리드 차원
 * @param blockDim 블록 차원
 * @param args 커널 인자 배열
 * @param sharedMem 동적 공유 메모리 크기
 * @param stream 스트림 핸들 (NULL이면 기본 스트림)
 * @return cbSuccess 성공
 */
cbError_t cbLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                         void **args, size_t sharedMem, cbStream_t stream);

/* ========== 에러 처리 ========== */

/**
 * 마지막 에러 가져오기 (에러 상태 리셋)
 *
 * @return 마지막 에러 코드
 */
cbError_t cbGetLastError(void);

/**
 * 마지막 에러 확인 (에러 상태 유지)
 *
 * @return 마지막 에러 코드
 */
cbError_t cbPeekAtLastError(void);

/**
 * 에러 이름 문자열
 *
 * @param error 에러 코드
 * @return 에러 이름 문자열
 */
const char* cbGetErrorName(cbError_t error);

/**
 * 에러 설명 문자열
 *
 * @param error 에러 코드
 * @return 에러 설명 문자열
 */
const char* cbGetErrorString(cbError_t error);

/* ========== eGPU/USB4 전용 기능 ========== */

/**
 * eGPU 연결 정보 조회
 *
 * @param info 연결 정보 구조체 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetConnectionInfo(cbConnectionInfo *info);

/**
 * USB4 대역폭 조회
 *
 * @param upBw 업스트림 대역폭 (bytes/s) 출력
 * @param downBw 다운스트림 대역폭 (bytes/s) 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetUSB4Bandwidth(size_t *upBw, size_t *downBw);

/**
 * eGPU 핫플러그 콜백 등록
 *
 * @param callback 콜백 함수 (device, connected)
 * @return cbSuccess 성공
 */
cbError_t cbSetHotplugCallback(void (*callback)(int device, int connected));

/**
 * PCIe 터널 상태 조회
 *
 * @param isActive 터널 활성화 상태 출력
 * @param bandwidth 할당된 대역폭 (Mbps) 출력
 * @return cbSuccess 성공
 */
cbError_t cbGetPCIeTunnelStatus(int *isActive, int *bandwidth);

#ifdef __cplusplus
}
#endif

/* ========== CUDA 호환 모드 ========== */
#ifdef CUDABRIDGE_COMPAT_MODE

/* CUDA API 이름을 CudaBridge API로 매핑 */
#define cudaSuccess             cbSuccess
#define cudaError_t             cbError_t

#define cudaGetDeviceCount      cbGetDeviceCount
#define cudaGetDevice           cbGetDevice
#define cudaSetDevice           cbSetDevice
#define cudaGetDeviceProperties(p,d) ({ \
    cbDeviceProp _prop; \
    cbError_t _err = cbGetDeviceProperties(&_prop, d); \
    if (_err == cbSuccess) memcpy(p, &_prop, sizeof(cbDeviceProp)); \
    _err; \
})
#define cudaDeviceSynchronize   cbDeviceSynchronize
#define cudaDeviceReset         cbDeviceReset

#define cudaMalloc              cbMalloc
#define cudaFree                cbFree
#define cudaMemcpy(d,s,c,k)     cbMemcpy(d, s, c, (cbMemcpyKind)(k))
#define cudaMemcpyAsync(d,s,c,k,st) cbMemcpyAsync(d, s, c, (cbMemcpyKind)(k), st)
#define cudaMemset              cbMemset
#define cudaMallocHost          cbMallocHost
#define cudaFreeHost            cbFreeHost
#define cudaMallocManaged(p,s,f) cbMallocManaged(p, s)
#define cudaMemGetInfo          cbMemGetInfo

#define cudaStreamCreate        cbStreamCreate
#define cudaStreamDestroy       cbStreamDestroy
#define cudaStreamSynchronize   cbStreamSynchronize

#define cudaEventCreate         cbEventCreate
#define cudaEventDestroy        cbEventDestroy
#define cudaEventRecord         cbEventRecord
#define cudaEventSynchronize    cbEventSynchronize
#define cudaEventElapsedTime    cbEventElapsedTime

#define cudaGetLastError        cbGetLastError
#define cudaPeekAtLastError     cbPeekAtLastError
#define cudaGetErrorName        cbGetErrorName
#define cudaGetErrorString      cbGetErrorString

#define cudaMemcpyHostToHost    CB_MEMCPY_HOST_TO_HOST
#define cudaMemcpyHostToDevice  CB_MEMCPY_HOST_TO_DEVICE
#define cudaMemcpyDeviceToHost  CB_MEMCPY_DEVICE_TO_HOST
#define cudaMemcpyDeviceToDevice CB_MEMCPY_DEVICE_TO_DEVICE

#endif /* CUDABRIDGE_COMPAT_MODE */

#endif /* CUDABRIDGE_H */
