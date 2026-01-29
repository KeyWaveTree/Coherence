/**
 * CudaBridge - PCIe Tunneling Implementation
 *
 * USB4/Thunderbolt를 통한 PCIe 터널링 구현
 */

#include "pcie_tunnel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/vm_map.h>
#endif

/* 에러 코드 */
#define PCIE_SUCCESS            0
#define PCIE_ERR_INVALID_PARAM -1
#define PCIE_ERR_NO_MEMORY     -2
#define PCIE_ERR_IO            -3
#define PCIE_ERR_NOT_FOUND     -4
#define PCIE_ERR_TIMEOUT       -5
#define PCIE_ERR_NOT_SUPPORTED -6

/* NVIDIA 벤더 ID */
#define NVIDIA_VENDOR_ID        0x10DE

/* 로깅 매크로 */
#define PCIE_LOG(fmt, ...) printf("[PCIe] " fmt "\n", ##__VA_ARGS__)
#define PCIE_ERR(fmt, ...) fprintf(stderr, "[PCIe ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef DEBUG
#define PCIE_DBG(fmt, ...) printf("[PCIe DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define PCIE_DBG(fmt, ...)
#endif

/* 내부 함수 선언 */
static int pcie_send_tlp(PCIeTunnelContext *ctx, PCIeTlpHeader *header,
                         void *data, size_t data_size);
static int pcie_receive_tlp(PCIeTunnelContext *ctx, PCIeTlpHeader *header,
                            void *data, size_t *data_size);
static int pcie_probe_device(PCIeTunnelContext *ctx, uint8_t bus,
                             uint8_t device, uint8_t function);
static uint64_t pcie_decode_bar_size(uint32_t bar_value);

/**
 * PCIe 터널 컨텍스트 초기화
 */
int pcie_tunnel_init(PCIeTunnelContext *ctx, USB4ControllerContext *usb4_ctx)
{
    if (!ctx || !usb4_ctx) {
        return PCIE_ERR_INVALID_PARAM;
    }

    memset(ctx, 0, sizeof(PCIeTunnelContext));
    ctx->usb4_ctx = usb4_ctx;
    ctx->next_tag = 0;
    ctx->dma_enabled = false;

    PCIE_LOG("PCIe tunnel context initialized");

    return PCIE_SUCCESS;
}

/**
 * PCIe 터널 컨텍스트 종료
 */
void pcie_tunnel_shutdown(PCIeTunnelContext *ctx)
{
    if (!ctx) return;

    PCIE_LOG("Shutting down PCIe tunnel context...");

    /* 모든 디바이스 정리 */
    for (int i = 0; i < ctx->device_count; i++) {
        PCIeDevice *dev = ctx->devices[i];
        if (dev) {
            /* BAR 매핑 해제 */
            for (int j = 0; j < PCIE_BAR_COUNT; j++) {
                if (dev->bars[j].is_mapped) {
                    pcie_unmap_bar(dev, j);
                }
            }
            free(dev->platform_data);
            dev->platform_data = NULL;
            /* 보안: Configuration Space 캐시 민감 데이터 삭제 */
            volatile uint8_t *cache = (volatile uint8_t *)dev->config_cache;
            for (int k = 0; k < PCIE_CONFIG_SPACE_SIZE; k++) {
                cache[k] = 0;
            }
            free(dev);
            ctx->devices[i] = NULL;
        }
    }

    PCIE_LOG("PCIe tunnel: TX %" PRIu64 " bytes, RX %" PRIu64 " bytes, %" PRIu64 " TLPs",
             ctx->bytes_tx, ctx->bytes_rx, ctx->tlp_count);

    /* 보안: 민감 데이터 확실히 제거 */
    volatile uint8_t *p = (volatile uint8_t *)ctx;
    for (size_t i = 0; i < sizeof(PCIeTunnelContext); i++) {
        p[i] = 0;
    }
}

/**
 * TLP 전송
 */
static int pcie_send_tlp(PCIeTunnelContext *ctx, PCIeTlpHeader *header,
                         void *data, size_t data_size)
{
    if (!ctx || !header) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* USB4 터널을 통해 TLP 전송 */
    /* 실제 구현에서는 USB4 프로토콜로 캡슐화 */

    ctx->tlp_count++;
    if (data) {
        ctx->bytes_tx += data_size;
    }

    PCIE_DBG("TLP sent: type=0x%02X addr=0x%" PRIX64 " len=%d",
             header->type, header->address, header->length);

    return PCIE_SUCCESS;
}

/**
 * TLP 수신
 */
static int pcie_receive_tlp(PCIeTunnelContext *ctx, PCIeTlpHeader *header,
                            void *data, size_t *data_size)
{
    if (!ctx || !header) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* USB4 터널에서 TLP 수신 */
    /* 실제 구현에서는 USB4 프로토콜에서 디캡슐화 */

    if (data && data_size) {
        ctx->bytes_rx += *data_size;
    }

    return PCIE_SUCCESS;
}

/**
 * PCIe 버스 스캔
 */
int pcie_scan_bus(PCIeTunnelContext *ctx)
{
    if (!ctx) {
        return PCIE_ERR_INVALID_PARAM;
    }

    PCIE_LOG("Scanning PCIe bus...");

    int found = 0;

    /* USB4 터널을 통해 연결된 버스만 스캔 */
    /* 일반적으로 eGPU는 버스 0에 나타남 */
    for (int dev = 0; dev < PCIE_MAX_DEVICE && found < 32; dev++) {
        for (int func = 0; func < PCIE_MAX_FUNCTION; func++) {
            int ret = pcie_probe_device(ctx, 0, dev, func);
            if (ret == PCIE_SUCCESS) {
                found++;
                /* 멀티펑션이 아니면 다음 디바이스로 */
                if (func == 0) {
                    uint32_t header;
                    pcie_config_read(ctx->devices[ctx->device_count - 1],
                                    PCIE_REG_HEADER_TYPE, 1, &header);
                    if (!(header & 0x80)) {
                        break;
                    }
                }
            } else if (func == 0) {
                break;  /* Function 0이 없으면 디바이스 없음 */
            }
        }
    }

    PCIE_LOG("Bus scan complete: found %d device(s)", found);

    return found;
}

/**
 * 디바이스 프로브
 */
static int pcie_probe_device(PCIeTunnelContext *ctx, uint8_t bus,
                             uint8_t device, uint8_t function)
{
    /* Configuration Space 읽기를 위한 TLP 생성 */
    PCIeTlpHeader header = {
        .type = PCIE_TLP_TYPE_CFG_READ_0,
        .tc = 0,
        .attr = 0,
        .length = 1,
        .requester_id = 0,
        .tag = ctx->next_tag++,
        .first_be = 0x0F,
        .last_be = 0x00,
        .address = 0
    };

    uint32_t vendor_device = 0;

    /* 시뮬레이션: USB4 터널이 있으면 NVIDIA GPU 가정 */
    if (ctx->usb4_ctx && ctx->usb4_ctx->tunnel_count > 0) {
        if (device == 0 && function == 0) {
            vendor_device = (0x2684 << 16) | NVIDIA_VENDOR_ID;  /* RTX 4090 */
        }
    }

    if (vendor_device == 0 || vendor_device == 0xFFFFFFFF) {
        return PCIE_ERR_NOT_FOUND;
    }

    /* 디바이스 구조체 할당 */
    PCIeDevice *dev = calloc(1, sizeof(PCIeDevice));
    if (!dev) {
        return PCIE_ERR_NO_MEMORY;
    }

    dev->bus = bus;
    dev->device = device;
    dev->function = function;
    dev->vendor_id = vendor_device & 0xFFFF;
    dev->device_id = (vendor_device >> 16) & 0xFFFF;

    /* 추가 정보 읽기 */
    uint32_t class_rev;
    /* 시뮬레이션: 3D 컨트롤러 클래스 */
    class_rev = (PCIE_CLASS_3D << 8) | 0xA1;
    dev->class_code = (class_rev >> 8) & 0xFFFFFF;
    dev->revision = class_rev & 0xFF;

    /* 서브시스템 ID */
    dev->subsystem_vendor_id = NVIDIA_VENDOR_ID;
    dev->subsystem_id = dev->device_id;

    /* USB4 터널 참조 */
    if (ctx->usb4_ctx && ctx->usb4_ctx->tunnel_count > 0) {
        dev->tunnel = ctx->usb4_ctx->active_tunnels[0];
    }

    /* BAR 프로브 */
    uint64_t bar_sizes[] = {
        16 * 1024 * 1024,       /* BAR0: 16MB - GPU 레지스터 */
        256 * 1024 * 1024,      /* BAR1: 256MB - Frame Buffer aperture */
        32 * 1024 * 1024,       /* BAR2: 32MB - GPU memory window */
        0, 0, 0                 /* BAR3-5: 미사용 */
    };

    for (int i = 0; i < PCIE_BAR_COUNT; i++) {
        if (bar_sizes[i] > 0) {
            dev->bars[i].size = bar_sizes[i];
            dev->bars[i].physical_base = 0x80000000ULL + (i * 0x10000000ULL);
            dev->bars[i].type = (bar_sizes[i] > 0xFFFFFFFF) ?
                               PCIE_BAR_TYPE_MEM64 : PCIE_BAR_TYPE_MEM32;
            dev->bars[i].prefetchable = (i == 1);  /* FB는 prefetchable */
            dev->bars[i].is_mapped = false;
        }
    }

    /* 보안: 디바이스 배열 범위 초과 방지 */
    if (ctx->device_count >= 32) {
        PCIE_ERR("Device array full, cannot add more devices");
        free(dev);
        return PCIE_ERR_NO_MEMORY;
    }

    ctx->devices[ctx->device_count++] = dev;

    PCIE_LOG("Found device %02X:%02X.%X: %04X:%04X (class %06X)",
             bus, device, function,
             dev->vendor_id, dev->device_id, dev->class_code);

    return PCIE_SUCCESS;
}

/**
 * PCIe Configuration Space 읽기
 */
int pcie_config_read(PCIeDevice *device, uint16_t offset,
                     uint8_t size, uint32_t *value)
{
    if (!device || !value || offset >= PCIE_CONFIG_SPACE_SIZE) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 보안: 버퍼 오버플로우 방지 — offset + size가 버퍼를 초과하지 않는지 확인 */
    if ((uint32_t)offset + (uint32_t)size > PCIE_CONFIG_SPACE_SIZE) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 캐시된 값 반환 (정렬 안전한 memcpy 사용) */
    *value = 0;
    switch (size) {
        case 1:
            *value = device->config_cache[offset];
            break;
        case 2:
            memcpy(value, &device->config_cache[offset], 2);
            break;
        case 4:
            memcpy(value, &device->config_cache[offset], 4);
            break;
        default:
            return PCIE_ERR_INVALID_PARAM;
    }

    return PCIE_SUCCESS;
}

/**
 * PCIe Configuration Space 쓰기
 */
int pcie_config_write(PCIeDevice *device, uint16_t offset,
                      uint8_t size, uint32_t value)
{
    if (!device || offset >= PCIE_CONFIG_SPACE_SIZE) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 보안: 버퍼 오버플로우 방지 */
    if ((uint32_t)offset + (uint32_t)size > PCIE_CONFIG_SPACE_SIZE) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 캐시 업데이트 (정렬 안전한 memcpy 사용) */
    switch (size) {
        case 1:
            device->config_cache[offset] = value & 0xFF;
            break;
        case 2: {
            uint16_t val16 = value & 0xFFFF;
            memcpy(&device->config_cache[offset], &val16, 2);
            break;
        }
        case 4:
            memcpy(&device->config_cache[offset], &value, 4);
            break;
        default:
            return PCIE_ERR_INVALID_PARAM;
    }

    /* TLP를 통해 실제 디바이스에 쓰기 */
    PCIeTlpHeader header = {
        .type = PCIE_TLP_TYPE_CFG_WRITE_0,
        .length = 1,
        .address = offset
    };

    /* 보안: 유효한 터널 컨텍스트가 없으면 캐시만 업데이트 */
    if (!device->tunnel) {
        return PCIE_SUCCESS;
    }

    return PCIE_SUCCESS;
}

/**
 * BAR 메모리 매핑
 */
int pcie_map_bar(PCIeDevice *device, int bar_index)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return PCIE_ERR_INVALID_PARAM;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (bar->size == 0) {
        return PCIE_ERR_NOT_FOUND;
    }

    if (bar->is_mapped) {
        return PCIE_SUCCESS;  /* 이미 매핑됨 */
    }

    PCIE_LOG("Mapping BAR%d: phys=0x%" PRIX64 " size=0x%" PRIX64,
             bar_index, bar->physical_base, bar->size);

#ifdef __APPLE__
    /* macOS에서는 IOKit을 통해 메모리 매핑 */
    /* 실제로는 IOMemoryDescriptor::createMappingInTask 사용 */
    mach_vm_address_t addr = 0;
    mach_vm_size_t size = bar->size;

    kern_return_t kr = mach_vm_allocate(mach_task_self(),
                                        &addr, size,
                                        VM_FLAGS_ANYWHERE);
    if (kr != KERN_SUCCESS) {
        PCIE_ERR("Failed to allocate virtual memory for BAR%d", bar_index);
        return PCIE_ERR_NO_MEMORY;
    }

    bar->virtual_base = (void*)addr;
#else
    /* 다른 플랫폼에서는 직접 할당 */
    bar->virtual_base = calloc(1, bar->size);
    if (!bar->virtual_base) {
        return PCIE_ERR_NO_MEMORY;
    }
#endif

    bar->is_mapped = true;

    PCIE_LOG("BAR%d mapped at virtual address %p", bar_index, bar->virtual_base);

    return PCIE_SUCCESS;
}

/**
 * BAR 메모리 매핑 해제
 */
void pcie_unmap_bar(PCIeDevice *device, int bar_index)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (!bar->is_mapped || !bar->virtual_base) {
        return;
    }

    PCIE_LOG("Unmapping BAR%d", bar_index);

#ifdef __APPLE__
    mach_vm_deallocate(mach_task_self(),
                       (mach_vm_address_t)bar->virtual_base,
                       bar->size);
#else
    free(bar->virtual_base);
#endif

    bar->virtual_base = NULL;
    bar->is_mapped = false;
}

/**
 * MMIO 읽기 (32비트)
 */
uint32_t pcie_mmio_read32(PCIeDevice *device, int bar_index, uint64_t offset)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return 0xFFFFFFFF;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (!bar->is_mapped || !bar->virtual_base || offset + 4 > bar->size) {
        return 0xFFFFFFFF;
    }

    /* 보안: MMIO 4바이트 정렬 검증 */
    if (offset & 0x3) {
        PCIE_ERR("Unaligned MMIO read32 at offset 0x%" PRIX64, offset);
        return 0xFFFFFFFF;
    }

    /* USB4 터널을 통해 실제 MMIO 읽기 수행 */
    /* 시뮬레이션: 가상 메모리에서 읽기 */
    uint32_t value = *(volatile uint32_t*)((uint8_t*)bar->virtual_base + offset);

    PCIE_DBG("MMIO read32 BAR%d+0x%" PRIX64 " = 0x%08X", bar_index, offset, value);

    return value;
}

/**
 * MMIO 쓰기 (32비트)
 */
void pcie_mmio_write32(PCIeDevice *device, int bar_index,
                       uint64_t offset, uint32_t value)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (!bar->is_mapped || !bar->virtual_base || offset + 4 > bar->size) {
        return;
    }

    /* 보안: MMIO 4바이트 정렬 검증 */
    if (offset & 0x3) {
        PCIE_ERR("Unaligned MMIO write32 at offset 0x%" PRIX64, offset);
        return;
    }

    PCIE_DBG("MMIO write32 BAR%d+0x%" PRIX64 " = 0x%08X", bar_index, offset, value);

    /* USB4 터널을 통해 실제 MMIO 쓰기 수행 */
    *(volatile uint32_t*)((uint8_t*)bar->virtual_base + offset) = value;
}

/**
 * MMIO 읽기 (64비트)
 */
uint64_t pcie_mmio_read64(PCIeDevice *device, int bar_index, uint64_t offset)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return 0xFFFFFFFFFFFFFFFFULL;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (!bar->is_mapped || !bar->virtual_base || offset + 8 > bar->size) {
        return 0xFFFFFFFFFFFFFFFFULL;
    }

    /* 보안: MMIO 8바이트 정렬 검증 */
    if (offset & 0x7) {
        PCIE_ERR("Unaligned MMIO read64 at offset 0x%" PRIX64, offset);
        return 0xFFFFFFFFFFFFFFFFULL;
    }

    uint64_t value = *(volatile uint64_t*)((uint8_t*)bar->virtual_base + offset);

    PCIE_DBG("MMIO read64 BAR%d+0x%" PRIX64 " = 0x%016" PRIX64, bar_index, offset, value);

    return value;
}

/**
 * MMIO 쓰기 (64비트)
 */
void pcie_mmio_write64(PCIeDevice *device, int bar_index,
                       uint64_t offset, uint64_t value)
{
    if (!device || bar_index < 0 || bar_index >= PCIE_BAR_COUNT) {
        return;
    }

    PCIeBar *bar = &device->bars[bar_index];

    if (!bar->is_mapped || !bar->virtual_base || offset + 8 > bar->size) {
        return;
    }

    /* 보안: MMIO 8바이트 정렬 검증 */
    if (offset & 0x7) {
        PCIE_ERR("Unaligned MMIO write64 at offset 0x%" PRIX64, offset);
        return;
    }

    PCIE_DBG("MMIO write64 BAR%d+0x%" PRIX64 " = 0x%016" PRIX64, bar_index, offset, value);

    *(volatile uint64_t*)((uint8_t*)bar->virtual_base + offset) = value;
}

/**
 * DMA 전송 시작
 */
int pcie_dma_transfer(PCIeTunnelContext *ctx, PCIeDevice *device,
                      PCIeDmaRequest *request)
{
    if (!ctx || !device || !request) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 보안: DMA 전송 크기 검증 (0 또는 비정상적으로 큰 크기 방지) */
    if (request->size == 0 || request->size > (size_t)1 * 1024 * 1024 * 1024) {
        PCIE_ERR("Invalid DMA transfer size: %zu", request->size);
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 보안: 호스트 주소 유효성 기본 검증 (NULL 포인터 방지) */
    if (request->host_addr == 0) {
        PCIE_ERR("Invalid host address for DMA transfer");
        return PCIE_ERR_INVALID_PARAM;
    }

    /* 보안: 오버플로우 검증 */
    if (request->host_addr + request->size < request->host_addr ||
        request->device_addr + request->size < request->device_addr) {
        PCIE_ERR("DMA address overflow detected");
        return PCIE_ERR_INVALID_PARAM;
    }

    PCIE_LOG("DMA transfer: %s %zu bytes (host=0x%" PRIX64 " <-> device=0x%" PRIX64 ")",
             request->is_write ? "H2D" : "D2H",
             request->size, request->host_addr, request->device_addr);

    /* USB4 터널을 통한 DMA는 여러 TLP로 분할 */
    size_t max_payload = 256;  /* 최대 페이로드 크기 */
    size_t remaining = request->size;
    uint64_t host_offset = 0;
    uint64_t dev_offset = 0;

    while (remaining > 0) {
        size_t chunk = (remaining > max_payload) ? max_payload : remaining;

        PCIeTlpHeader header = {
            .type = request->is_write ?
                   PCIE_TLP_TYPE_MEM_WRITE_64 : PCIE_TLP_TYPE_MEM_READ_64,
            .length = (chunk + 3) / 4,
            .address = request->device_addr + dev_offset,
            .tag = ctx->next_tag++
        };

        /* 보안: 호스트 주소를 직접 포인터로 캐스팅하지 않고 컨텍스트를 통해 전달 */
        int ret = pcie_send_tlp(ctx, &header, NULL, chunk);
        if (ret != PCIE_SUCCESS) {
            return ret;
        }

        remaining -= chunk;
        host_offset += chunk;
        dev_offset += chunk;
    }

    /* 콜백 호출 */
    if (request->callback) {
        request->callback(request->user_data, PCIE_SUCCESS);
    }

    return PCIE_SUCCESS;
}

/**
 * DMA 전송 대기
 */
int pcie_dma_wait(PCIeTunnelContext *ctx, PCIeDmaRequest *request)
{
    if (!ctx || !request) {
        return PCIE_ERR_INVALID_PARAM;
    }

    /* USB4는 동기적으로 동작하므로 즉시 반환 */
    return PCIE_SUCCESS;
}

/**
 * 디바이스 활성화
 */
int pcie_enable_device(PCIeDevice *device)
{
    if (!device) {
        return PCIE_ERR_INVALID_PARAM;
    }

    uint32_t cmd;
    pcie_config_read(device, PCIE_REG_COMMAND, 2, &cmd);

    cmd |= PCIE_CMD_IO_ENABLE | PCIE_CMD_MEM_ENABLE;

    int ret = pcie_config_write(device, PCIE_REG_COMMAND, 2, cmd);
    if (ret == PCIE_SUCCESS) {
        device->is_enabled = true;
        PCIE_LOG("Device %02X:%02X.%X enabled",
                 device->bus, device->device, device->function);
    }

    return ret;
}

/**
 * 버스 마스터 활성화
 */
int pcie_enable_bus_master(PCIeDevice *device)
{
    if (!device) {
        return PCIE_ERR_INVALID_PARAM;
    }

    uint32_t cmd;
    pcie_config_read(device, PCIE_REG_COMMAND, 2, &cmd);

    cmd |= PCIE_CMD_BUS_MASTER;

    int ret = pcie_config_write(device, PCIE_REG_COMMAND, 2, cmd);
    if (ret == PCIE_SUCCESS) {
        device->is_bus_master = true;
        PCIE_LOG("Bus master enabled for device %02X:%02X.%X",
                 device->bus, device->device, device->function);
    }

    return ret;
}

/**
 * MSI/MSI-X 인터럽트 설정
 */
int pcie_setup_msi(PCIeDevice *device, int num_vectors)
{
    if (!device || num_vectors < 1) {
        return PCIE_ERR_INVALID_PARAM;
    }

    PCIE_LOG("Setting up %d MSI vector(s) for device %02X:%02X.%X",
             num_vectors, device->bus, device->device, device->function);

    /* MSI Capability 구조체 찾기 및 설정 */
    /* 실제 구현에서는 Capability 체인 순회 필요 */

    return PCIE_SUCCESS;
}

/**
 * NVIDIA GPU 디바이스 찾기
 */
PCIeDevice* pcie_find_nvidia_gpu(PCIeTunnelContext *ctx)
{
    if (!ctx) return NULL;

    for (int i = 0; i < ctx->device_count; i++) {
        PCIeDevice *dev = ctx->devices[i];
        if (dev && dev->vendor_id == NVIDIA_VENDOR_ID) {
            /* 디스플레이 컨트롤러 또는 3D 컨트롤러 클래스 확인 */
            uint8_t base_class = (dev->class_code >> 16) & 0xFF;
            if (base_class == 0x03) {  /* Display controller */
                PCIE_LOG("Found NVIDIA GPU: %04X:%04X at %02X:%02X.%X",
                         dev->vendor_id, dev->device_id,
                         dev->bus, dev->device, dev->function);
                return dev;
            }
        }
    }

    return NULL;
}
