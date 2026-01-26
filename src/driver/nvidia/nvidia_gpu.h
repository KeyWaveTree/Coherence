/**
 * CudaBridge - NVIDIA GPU Driver Interface
 *
 * NVIDIA GPU 하드웨어와의 저수준 통신을 담당합니다.
 * GPU 초기화, 명령 제출, 메모리 관리 등을 처리합니다.
 */

#ifndef CUDABRIDGE_NVIDIA_GPU_H
#define CUDABRIDGE_NVIDIA_GPU_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../../kernel/pcie/pcie_tunnel.h"

/* NVIDIA GPU 아키텍처 */
typedef enum {
    NV_ARCH_UNKNOWN = 0,
    NV_ARCH_KEPLER = 0x0E0,      /* GK1xx */
    NV_ARCH_MAXWELL = 0x110,     /* GM1xx, GM2xx */
    NV_ARCH_PASCAL = 0x130,      /* GP1xx */
    NV_ARCH_VOLTA = 0x140,       /* GV1xx */
    NV_ARCH_TURING = 0x160,      /* TU1xx */
    NV_ARCH_AMPERE = 0x170,      /* GA1xx */
    NV_ARCH_ADA = 0x190,         /* AD1xx */
    NV_ARCH_HOPPER = 0x180,      /* GH1xx */
    NV_ARCH_BLACKWELL = 0x1A0   /* GB1xx */
} NVArchitecture;

/* GPU 상태 */
typedef enum {
    NV_GPU_STATE_UNKNOWN = 0,
    NV_GPU_STATE_UNINITIALIZED,
    NV_GPU_STATE_INITIALIZING,
    NV_GPU_STATE_READY,
    NV_GPU_STATE_BUSY,
    NV_GPU_STATE_ERROR,
    NV_GPU_STATE_SUSPENDED
} NVGpuState;

/* NVIDIA 레지스터 오프셋 (BAR0 기준) */
#define NV_PMC_BOOT_0           0x00000000  /* Boot register 0 */
#define NV_PMC_BOOT_1           0x00000004  /* Boot register 1 */
#define NV_PMC_ENABLE           0x00000200  /* Engine enable */
#define NV_PMC_INTR_0           0x00000100  /* Interrupt status */
#define NV_PMC_INTR_EN_0        0x00000140  /* Interrupt enable */

#define NV_PBUS_PCI_NV_0        0x00001800  /* PCI config space mirror */
#define NV_PBUS_PCI_NV_1        0x00001804

#define NV_PFIFO_RUNLIST        0x00002200  /* FIFO runlist */
#define NV_PFIFO_PB_BASE        0x00002270  /* Push buffer base */

#define NV_PGRAPH_STATUS        0x00400700  /* Graphics engine status */
#define NV_PGRAPH_TRAPPED_ADDR  0x00400704

#define NV_PFB_CFG0             0x00100200  /* Frame buffer config */
#define NV_PFB_NISO_CFG0        0x00100C14

/* 메모리 타입 */
typedef enum {
    NV_MEM_TYPE_SYSTEM = 0,     /* 시스템 메모리 */
    NV_MEM_TYPE_VIDEO,          /* 비디오 메모리 (VRAM) */
    NV_MEM_TYPE_PEER            /* 피어 메모리 */
} NVMemoryType;

/* 메모리 할당 정보 */
typedef struct NVMemoryAlloc {
    uint64_t            gpu_addr;       /* GPU 가상 주소 */
    uint64_t            host_addr;      /* 호스트 가상 주소 (매핑된 경우) */
    uint64_t            size;           /* 크기 */
    NVMemoryType        type;           /* 메모리 타입 */
    bool                is_mapped;      /* 호스트에 매핑됨 */
    uint32_t            flags;          /* 할당 플래그 */
    void               *backing_store;  /* 가상 GPU 모드용 backing store */
} NVMemoryAlloc;

/* 채널 정보 */
typedef struct NVChannel {
    uint32_t            id;             /* 채널 ID */
    uint64_t            pb_base;        /* Push buffer 베이스 주소 */
    uint64_t            pb_size;        /* Push buffer 크기 */
    uint32_t            pb_put;         /* PUT 포인터 */
    uint32_t            pb_get;         /* GET 포인터 */
    bool                is_active;      /* 활성화 상태 */
    void               *user_data;      /* 사용자 데이터 */
} NVChannel;

/* SM (Streaming Multiprocessor) 정보 */
typedef struct NVSmInfo {
    uint32_t            count;          /* SM 수 */
    uint32_t            cores_per_sm;   /* SM당 코어 수 */
    uint32_t            max_threads_per_sm;
    uint32_t            max_blocks_per_sm;
    uint32_t            shared_mem_per_sm;
    uint32_t            registers_per_sm;
} NVSmInfo;

/* GPU 디바이스 정보 */
typedef struct NVDeviceInfo {
    char                name[256];      /* GPU 이름 */
    NVArchitecture      architecture;   /* 아키텍처 */
    uint32_t            chip_id;        /* 칩 ID */
    uint32_t            revision;       /* 리비전 */

    /* 메모리 정보 */
    uint64_t            vram_size;      /* VRAM 크기 */
    uint32_t            vram_type;      /* VRAM 타입 (GDDR6 등) */
    uint32_t            vram_bus_width; /* VRAM 버스 폭 (비트) */
    uint64_t            bandwidth;      /* 메모리 대역폭 (bytes/sec) */

    /* 컴퓨트 정보 */
    NVSmInfo            sm_info;        /* SM 정보 */
    uint32_t            compute_cap_major;
    uint32_t            compute_cap_minor;
    uint32_t            max_threads_per_block;
    uint32_t            max_grid_dim[3];
    uint32_t            max_block_dim[3];

    /* 클럭 정보 */
    uint32_t            gpu_clock_mhz;
    uint32_t            mem_clock_mhz;
} NVDeviceInfo;

/* GPU 컨텍스트 */
typedef struct NVGpuContext {
    PCIeDevice         *pcie_device;    /* PCIe 디바이스 */
    NVDeviceInfo        info;           /* 디바이스 정보 */
    NVGpuState          state;          /* 현재 상태 */

    /* 채널 관리 */
    NVChannel          *channels[64];
    uint32_t            channel_count;

    /* 메모리 관리 */
    uint64_t            vram_free;      /* 사용 가능한 VRAM */
    NVMemoryAlloc      *allocations[4096];
    uint32_t            alloc_count;

    /* 통계 */
    uint64_t            kernel_launches;
    uint64_t            bytes_transferred;

    /* 락 (스레드 안전성) */
    void               *lock;
} NVGpuContext;

/* 커널 실행 매개변수 */
typedef struct NVKernelParams {
    uint64_t            func_addr;      /* 커널 함수 주소 */
    uint32_t            grid_dim[3];    /* 그리드 차원 */
    uint32_t            block_dim[3];   /* 블록 차원 */
    uint32_t            shared_mem;     /* 동적 공유 메모리 크기 */
    NVChannel          *channel;        /* 실행 채널 */
    void              **args;           /* 커널 인자 */
    size_t             *arg_sizes;      /* 인자 크기 */
    int                 arg_count;      /* 인자 수 */
} NVKernelParams;

/* 함수 선언 */

/**
 * GPU 초기화
 */
int nv_gpu_init(NVGpuContext *ctx, PCIeDevice *pcie_device);

/**
 * GPU 종료
 */
void nv_gpu_shutdown(NVGpuContext *ctx);

/**
 * GPU 리셋
 */
int nv_gpu_reset(NVGpuContext *ctx);

/**
 * 디바이스 정보 쿼리
 */
int nv_gpu_query_info(NVGpuContext *ctx);

/**
 * 레지스터 읽기
 */
uint32_t nv_gpu_read_reg(NVGpuContext *ctx, uint32_t offset);

/**
 * 레지스터 쓰기
 */
void nv_gpu_write_reg(NVGpuContext *ctx, uint32_t offset, uint32_t value);

/**
 * 채널 생성
 */
int nv_gpu_create_channel(NVGpuContext *ctx, NVChannel **channel);

/**
 * 채널 제거
 */
void nv_gpu_destroy_channel(NVGpuContext *ctx, NVChannel *channel);

/**
 * 메모리 할당
 */
int nv_gpu_alloc_memory(NVGpuContext *ctx, NVMemoryAlloc *alloc);

/**
 * 메모리 해제
 */
void nv_gpu_free_memory(NVGpuContext *ctx, NVMemoryAlloc *alloc);

/**
 * 메모리 복사 (호스트 → 디바이스)
 */
int nv_gpu_memcpy_h2d(NVGpuContext *ctx, uint64_t dst,
                      const void *src, size_t size);

/**
 * 메모리 복사 (디바이스 → 호스트)
 */
int nv_gpu_memcpy_d2h(NVGpuContext *ctx, void *dst,
                      uint64_t src, size_t size);

/**
 * 메모리 복사 (디바이스 → 디바이스)
 */
int nv_gpu_memcpy_d2d(NVGpuContext *ctx, uint64_t dst,
                      uint64_t src, size_t size);

/**
 * 커널 실행
 */
int nv_gpu_launch_kernel(NVGpuContext *ctx, NVKernelParams *params);

/**
 * 동기화 (모든 작업 완료 대기)
 */
int nv_gpu_synchronize(NVGpuContext *ctx);

/**
 * 채널 동기화
 */
int nv_gpu_channel_sync(NVGpuContext *ctx, NVChannel *channel);

/**
 * 에러 체크
 */
int nv_gpu_check_error(NVGpuContext *ctx);

/**
 * 전력 상태 설정
 */
int nv_gpu_set_power_state(NVGpuContext *ctx, int state);

/**
 * 온도 읽기
 */
int nv_gpu_get_temperature(NVGpuContext *ctx, int *temp_celsius);

/**
 * 클럭 속도 설정
 */
int nv_gpu_set_clocks(NVGpuContext *ctx, uint32_t gpu_mhz, uint32_t mem_mhz);

/**
 * 아키텍처 이름 반환
 */
const char* nv_arch_name(NVArchitecture arch);

#endif /* CUDABRIDGE_NVIDIA_GPU_H */
