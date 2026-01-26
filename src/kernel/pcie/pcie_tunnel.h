/**
 * CudaBridge - PCIe Tunneling Layer
 *
 * USB4/Thunderbolt를 통한 PCIe 패킷 터널링을 담당합니다.
 * PCIe Configuration Space, Memory-Mapped I/O, DMA를 처리합니다.
 */

#ifndef CUDABRIDGE_PCIE_TUNNEL_H
#define CUDABRIDGE_PCIE_TUNNEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../usb4/usb4_controller.h"

/* PCIe 상수 */
#define PCIE_MAX_BUS            256
#define PCIE_MAX_DEVICE         32
#define PCIE_MAX_FUNCTION       8
#define PCIE_CONFIG_SPACE_SIZE  4096
#define PCIE_BAR_COUNT          6

/* PCIe Configuration Space 레지스터 오프셋 */
#define PCIE_REG_VENDOR_ID      0x00
#define PCIE_REG_DEVICE_ID      0x02
#define PCIE_REG_COMMAND        0x04
#define PCIE_REG_STATUS         0x06
#define PCIE_REG_REVISION       0x08
#define PCIE_REG_CLASS_CODE     0x09
#define PCIE_REG_CACHE_LINE     0x0C
#define PCIE_REG_LATENCY        0x0D
#define PCIE_REG_HEADER_TYPE    0x0E
#define PCIE_REG_BIST           0x0F
#define PCIE_REG_BAR0           0x10
#define PCIE_REG_BAR1           0x14
#define PCIE_REG_BAR2           0x18
#define PCIE_REG_BAR3           0x1C
#define PCIE_REG_BAR4           0x20
#define PCIE_REG_BAR5           0x24
#define PCIE_REG_SUBSYSTEM_VID  0x2C
#define PCIE_REG_SUBSYSTEM_ID   0x2E
#define PCIE_REG_EXPROM_BASE    0x30
#define PCIE_REG_CAP_PTR        0x34
#define PCIE_REG_INT_LINE       0x3C
#define PCIE_REG_INT_PIN        0x3D

/* PCIe Command 레지스터 비트 */
#define PCIE_CMD_IO_ENABLE      (1 << 0)
#define PCIE_CMD_MEM_ENABLE     (1 << 1)
#define PCIE_CMD_BUS_MASTER     (1 << 2)
#define PCIE_CMD_SERR_ENABLE    (1 << 8)
#define PCIE_CMD_INT_DISABLE    (1 << 10)

/* BAR 타입 */
typedef enum {
    PCIE_BAR_TYPE_MEM32 = 0,
    PCIE_BAR_TYPE_MEM64 = 2,
    PCIE_BAR_TYPE_IO = 1
} PCIeBarType;

/* PCIe 디바이스 클래스 */
#define PCIE_CLASS_VGA          0x030000
#define PCIE_CLASS_3D           0x030200  /* NVIDIA GPU는 주로 이 클래스 */

/* PCIe BAR 구조체 */
typedef struct PCIeBar {
    uint64_t            physical_base;  /* 물리 주소 */
    uint64_t            size;           /* BAR 크기 */
    void               *virtual_base;   /* 매핑된 가상 주소 */
    PCIeBarType         type;           /* BAR 타입 */
    bool                prefetchable;   /* 프리페치 가능 여부 */
    bool                is_mapped;      /* 매핑 상태 */
} PCIeBar;

/* PCIe 디바이스 구조체 */
typedef struct PCIeDevice {
    /* BDF (Bus-Device-Function) */
    uint8_t             bus;
    uint8_t             device;
    uint8_t             function;

    /* 디바이스 식별 */
    uint16_t            vendor_id;
    uint16_t            device_id;
    uint16_t            subsystem_vendor_id;
    uint16_t            subsystem_id;
    uint32_t            class_code;
    uint8_t             revision;

    /* BAR 정보 */
    PCIeBar             bars[PCIE_BAR_COUNT];

    /* USB4 터널 참조 */
    USB4PCIeTunnel     *tunnel;

    /* 상태 */
    bool                is_enabled;
    bool                is_bus_master;

    /* Configuration Space 캐시 */
    uint8_t             config_cache[PCIE_CONFIG_SPACE_SIZE];

    /* 플랫폼별 데이터 */
    void               *platform_data;
} PCIeDevice;

/* PCIe 트랜잭션 타입 */
typedef enum {
    PCIE_TLP_TYPE_MEM_READ_32 = 0x00,
    PCIE_TLP_TYPE_MEM_READ_64 = 0x20,
    PCIE_TLP_TYPE_MEM_WRITE_32 = 0x40,
    PCIE_TLP_TYPE_MEM_WRITE_64 = 0x60,
    PCIE_TLP_TYPE_IO_READ = 0x02,
    PCIE_TLP_TYPE_IO_WRITE = 0x42,
    PCIE_TLP_TYPE_CFG_READ_0 = 0x04,
    PCIE_TLP_TYPE_CFG_WRITE_0 = 0x44,
    PCIE_TLP_TYPE_CFG_READ_1 = 0x05,
    PCIE_TLP_TYPE_CFG_WRITE_1 = 0x45,
    PCIE_TLP_TYPE_MSG = 0x30,
    PCIE_TLP_TYPE_CPL = 0x0A,
    PCIE_TLP_TYPE_CPL_DATA = 0x4A
} PCIeTlpType;

/* PCIe TLP (Transaction Layer Packet) 헤더 */
typedef struct PCIeTlpHeader {
    uint8_t             type;
    uint8_t             tc;             /* Traffic Class */
    uint8_t             attr;           /* Attributes */
    uint16_t            length;         /* DW 단위 */
    uint16_t            requester_id;
    uint8_t             tag;
    uint8_t             last_be;
    uint8_t             first_be;
    uint64_t            address;
} PCIeTlpHeader;

/* DMA 요청 구조체 */
typedef struct PCIeDmaRequest {
    uint64_t            host_addr;      /* 호스트 메모리 주소 */
    uint64_t            device_addr;    /* 디바이스 메모리 주소 */
    size_t              size;           /* 전송 크기 */
    bool                is_write;       /* true: H2D, false: D2H */
    void              (*callback)(void *user_data, int status);
    void               *user_data;
} PCIeDmaRequest;

/* PCIe 터널 컨텍스트 */
typedef struct PCIeTunnelContext {
    USB4ControllerContext *usb4_ctx;    /* USB4 컨트롤러 컨텍스트 */
    PCIeDevice         *devices[32];    /* 발견된 디바이스들 */
    uint8_t             device_count;   /* 디바이스 수 */

    /* TLP 태그 관리 */
    uint8_t             next_tag;

    /* DMA 엔진 */
    bool                dma_enabled;

    /* 통계 */
    uint64_t            bytes_tx;
    uint64_t            bytes_rx;
    uint64_t            tlp_count;
} PCIeTunnelContext;

/* 함수 선언 */

/**
 * PCIe 터널 컨텍스트 초기화
 */
int pcie_tunnel_init(PCIeTunnelContext *ctx, USB4ControllerContext *usb4_ctx);

/**
 * PCIe 터널 컨텍스트 종료
 */
void pcie_tunnel_shutdown(PCIeTunnelContext *ctx);

/**
 * PCIe 버스 스캔
 */
int pcie_scan_bus(PCIeTunnelContext *ctx);

/**
 * PCIe Configuration Space 읽기
 */
int pcie_config_read(PCIeDevice *device, uint16_t offset,
                     uint8_t size, uint32_t *value);

/**
 * PCIe Configuration Space 쓰기
 */
int pcie_config_write(PCIeDevice *device, uint16_t offset,
                      uint8_t size, uint32_t value);

/**
 * BAR 메모리 매핑
 */
int pcie_map_bar(PCIeDevice *device, int bar_index);

/**
 * BAR 메모리 매핑 해제
 */
void pcie_unmap_bar(PCIeDevice *device, int bar_index);

/**
 * MMIO 읽기 (32비트)
 */
uint32_t pcie_mmio_read32(PCIeDevice *device, int bar_index, uint64_t offset);

/**
 * MMIO 쓰기 (32비트)
 */
void pcie_mmio_write32(PCIeDevice *device, int bar_index,
                       uint64_t offset, uint32_t value);

/**
 * MMIO 읽기 (64비트)
 */
uint64_t pcie_mmio_read64(PCIeDevice *device, int bar_index, uint64_t offset);

/**
 * MMIO 쓰기 (64비트)
 */
void pcie_mmio_write64(PCIeDevice *device, int bar_index,
                       uint64_t offset, uint64_t value);

/**
 * DMA 전송 시작
 */
int pcie_dma_transfer(PCIeTunnelContext *ctx, PCIeDevice *device,
                      PCIeDmaRequest *request);

/**
 * DMA 전송 대기
 */
int pcie_dma_wait(PCIeTunnelContext *ctx, PCIeDmaRequest *request);

/**
 * 디바이스 활성화
 */
int pcie_enable_device(PCIeDevice *device);

/**
 * 버스 마스터 활성화
 */
int pcie_enable_bus_master(PCIeDevice *device);

/**
 * MSI/MSI-X 인터럽트 설정
 */
int pcie_setup_msi(PCIeDevice *device, int num_vectors);

/**
 * NVIDIA GPU 디바이스 찾기
 */
PCIeDevice* pcie_find_nvidia_gpu(PCIeTunnelContext *ctx);

#endif /* CUDABRIDGE_PCIE_TUNNEL_H */
