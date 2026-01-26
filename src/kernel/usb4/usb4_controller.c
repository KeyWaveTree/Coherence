/**
 * CudaBridge - USB4 Controller Implementation
 *
 * Apple Silicon USB4 컨트롤러 드라이버 구현
 */

#include "usb4_controller.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <Availability.h>

/* kIOMasterPortDefault was deprecated in macOS 12.0, use kIOMainPortDefault */
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 120000
#define CB_IO_MAIN_PORT kIOMainPortDefault
#else
#define CB_IO_MAIN_PORT kIOMasterPortDefault
#endif

#endif

/* 내부 에러 코드 */
#define USB4_SUCCESS                0
#define USB4_ERR_NOT_FOUND         -1
#define USB4_ERR_NO_MEMORY         -2
#define USB4_ERR_IO                -3
#define USB4_ERR_INVALID_PARAM     -4
#define USB4_ERR_NOT_SUPPORTED     -5
#define USB4_ERR_TIMEOUT           -6
#define USB4_ERR_TUNNEL_EXISTS     -7
#define USB4_ERR_NO_BANDWIDTH      -8

/* Apple Silicon USB4 컨트롤러 IOKit 클래스명 */
#define APPLE_USB4_CONTROLLER_CLASS "AppleUSB4Controller"
#define APPLE_TB_HOST_CONTROLLER    "AppleThunderboltHAL"

/* 로깅 매크로 */
#define USB4_LOG(fmt, ...) printf("[USB4] " fmt "\n", ##__VA_ARGS__)
#define USB4_ERR(fmt, ...) fprintf(stderr, "[USB4 ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef DEBUG
#define USB4_DBG(fmt, ...) printf("[USB4 DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define USB4_DBG(fmt, ...)
#endif

/* 내부 함수 선언 */
static int usb4_init_host_router(USB4ControllerContext *ctx);
static int usb4_enumerate_adapters(USB4Router *router);
static USB4Adapter* usb4_find_pcie_adapter(USB4Router *router, bool upstream);
static int usb4_configure_path(USB4PCIeTunnel *tunnel);
static int usb4_activate_tunnel(USB4PCIeTunnel *tunnel);

/**
 * USB4 컨트롤러 초기화
 */
int usb4_controller_init(USB4ControllerContext *ctx)
{
    if (!ctx) {
        return USB4_ERR_INVALID_PARAM;
    }

    memset(ctx, 0, sizeof(USB4ControllerContext));

    USB4_LOG("Initializing USB4 controller...");

#ifdef __APPLE__
    /* IOKit 서비스 찾기 */
    CFMutableDictionaryRef matching = IOServiceMatching(APPLE_USB4_CONTROLLER_CLASS);
    if (!matching) {
        /* Thunderbolt HAL로 폴백 */
        matching = IOServiceMatching(APPLE_TB_HOST_CONTROLLER);
        if (!matching) {
            USB4_ERR("Failed to create matching dictionary");
            return USB4_ERR_NOT_FOUND;
        }
    }

    io_service_t service = IOServiceGetMatchingService(CB_IO_MAIN_PORT, matching);
    if (!service) {
        USB4_ERR("USB4/Thunderbolt controller not found");
        return USB4_ERR_NOT_FOUND;
    }

    ctx->service = service;

    /* 연결 열기 */
    kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0, &ctx->connection);
    if (kr != KERN_SUCCESS) {
        USB4_ERR("Failed to open connection to USB4 controller: 0x%x", kr);
        IOObjectRelease(service);
        return USB4_ERR_IO;
    }

    USB4_LOG("Connected to USB4 controller service");
#endif

    /* 호스트 라우터 초기화 */
    int ret = usb4_init_host_router(ctx);
    if (ret != USB4_SUCCESS) {
        USB4_ERR("Failed to initialize host router");
        usb4_controller_shutdown(ctx);
        return ret;
    }

    USB4_LOG("USB4 controller initialized successfully");
    return USB4_SUCCESS;
}

/**
 * USB4 컨트롤러 종료
 */
void usb4_controller_shutdown(USB4ControllerContext *ctx)
{
    if (!ctx) return;

    USB4_LOG("Shutting down USB4 controller...");

    /* 모든 활성 터널 제거 */
    for (int i = 0; i < ctx->tunnel_count; i++) {
        if (ctx->active_tunnels[i]) {
            usb4_destroy_pcie_tunnel(ctx, ctx->active_tunnels[i]);
        }
    }

    /* 연결된 라우터 정리 */
    for (int i = 0; i < ctx->router_count; i++) {
        if (ctx->connected_routers[i]) {
            free(ctx->connected_routers[i]->platform_data);
            free(ctx->connected_routers[i]);
        }
    }

#ifdef __APPLE__
    if (ctx->connection) {
        IOServiceClose(ctx->connection);
    }
    if (ctx->service) {
        IOObjectRelease(ctx->service);
    }
#endif

    memset(ctx, 0, sizeof(USB4ControllerContext));
    USB4_LOG("USB4 controller shutdown complete");
}

/**
 * 호스트 라우터 초기화
 */
static int usb4_init_host_router(USB4ControllerContext *ctx)
{
    USB4Router *router = &ctx->host_router;

    router->route = 0;  /* 호스트는 route 0 */
    router->type = USB4_ROUTER_TYPE_HOST;
    router->is_connected = true;

#ifdef __APPLE__
    /* Apple Silicon 정보 읽기 */
    CFTypeRef prop;

    prop = IORegistryEntryCreateCFProperty(ctx->service,
                                           CFSTR("vendor-id"),
                                           kCFAllocatorDefault, 0);
    if (prop) {
        CFNumberGetValue(prop, kCFNumberIntType, &router->vendor_id);
        CFRelease(prop);
    } else {
        router->vendor_id = 0x106B;  /* Apple */
    }

    prop = IORegistryEntryCreateCFProperty(ctx->service,
                                           CFSTR("device-id"),
                                           kCFAllocatorDefault, 0);
    if (prop) {
        CFNumberGetValue(prop, kCFNumberIntType, &router->device_id);
        CFRelease(prop);
    }
#endif

    USB4_LOG("Host router initialized: vendor=0x%04X device=0x%04X",
             router->vendor_id, router->device_id);

    /* 어댑터 열거 */
    return usb4_enumerate_adapters(router);
}

/**
 * 라우터의 어댑터 열거
 */
static int usb4_enumerate_adapters(USB4Router *router)
{
    /* 실제 하드웨어 열거 대신 일반적인 Apple Silicon 구성 가정 */
    /* M1/M2/M3 칩은 보통 2-4개의 USB4 포트를 가짐 */

    router->adapter_count = 4;  /* 기본값 */

    USB4_DBG("Enumerated %d adapters on router 0x%" PRIX64,
             router->adapter_count, router->route);

    return USB4_SUCCESS;
}

/**
 * 연결된 라우터 스캔
 */
int usb4_scan_routers(USB4ControllerContext *ctx)
{
    if (!ctx) {
        return USB4_ERR_INVALID_PARAM;
    }

    USB4_LOG("Scanning for connected USB4 routers...");

    int found = 0;

#ifdef __APPLE__
    /* IOKit을 통해 연결된 Thunderbolt/USB4 장치 검색 */
    CFMutableDictionaryRef matching = IOServiceMatching("IOThunderboltDevice");
    if (!matching) {
        USB4_ERR("Failed to create matching dictionary for devices");
        return 0;
    }

    io_iterator_t iterator;
    kern_return_t kr = IOServiceGetMatchingServices(CB_IO_MAIN_PORT,
                                                    matching,
                                                    &iterator);
    if (kr != KERN_SUCCESS) {
        USB4_ERR("Failed to get matching services: 0x%x", kr);
        return 0;
    }

    io_service_t device;
    while ((device = IOIteratorNext(iterator)) && found < USB4_MAX_TUNNELS) {
        /* 장치 정보 읽기 */
        USB4Router *router = calloc(1, sizeof(USB4Router));
        if (!router) {
            IOObjectRelease(device);
            continue;
        }

        router->type = USB4_ROUTER_TYPE_DEVICE;
        router->is_connected = true;

        /* 벤더/디바이스 ID 읽기 */
        CFTypeRef prop = IORegistryEntryCreateCFProperty(device,
                                                         CFSTR("Vendor ID"),
                                                         kCFAllocatorDefault, 0);
        if (prop) {
            CFNumberGetValue(prop, kCFNumberIntType, &router->vendor_id);
            CFRelease(prop);
        }

        prop = IORegistryEntryCreateCFProperty(device,
                                               CFSTR("Device ID"),
                                               kCFAllocatorDefault, 0);
        if (prop) {
            CFNumberGetValue(prop, kCFNumberIntType, &router->device_id);
            CFRelease(prop);
        }

        /* 플랫폼 데이터 저장 */
        io_service_t *platform_data = malloc(sizeof(io_service_t));
        *platform_data = device;
        router->platform_data = platform_data;

        /* 어댑터 열거 */
        usb4_enumerate_adapters(router);

        ctx->connected_routers[found] = router;
        found++;

        USB4_LOG("Found USB4 device: vendor=0x%04X device=0x%04X",
                 router->vendor_id, router->device_id);
    }

    IOObjectRelease(iterator);
#endif

    ctx->router_count = found;
    USB4_LOG("Scan complete: found %d device(s)", found);

    return found;
}

/**
 * PCIe 어댑터 찾기
 */
static USB4Adapter* usb4_find_pcie_adapter(USB4Router *router, bool upstream)
{
    USB4Adapter *adapter = calloc(1, sizeof(USB4Adapter));
    if (!adapter) return NULL;

    adapter->router = router;
    adapter->type = upstream ? USB4_ADAPTER_PCIE_UP : USB4_ADAPTER_PCIE_DOWN;
    adapter->is_active = false;
    adapter->max_bandwidth = USB4_PCIE_TUNNEL_BW_GBPS * 1000;  /* Mbps */

    /* 어댑터 번호 설정 (실제로는 하드웨어에서 읽어야 함) */
    adapter->adapter_num = upstream ? 1 : 2;

    return adapter;
}

/**
 * PCIe 터널 생성
 */
int usb4_create_pcie_tunnel(USB4ControllerContext *ctx,
                            USB4Router *router,
                            USB4PCIeTunnel **tunnel_out)
{
    if (!ctx || !router || !tunnel_out) {
        return USB4_ERR_INVALID_PARAM;
    }

    if (ctx->tunnel_count >= USB4_MAX_TUNNELS) {
        USB4_ERR("Maximum tunnel count reached");
        return USB4_ERR_NO_BANDWIDTH;
    }

    USB4_LOG("Creating PCIe tunnel to router 0x%" PRIX64 "...", router->route);

    /* 터널 구조체 할당 */
    USB4PCIeTunnel *tunnel = calloc(1, sizeof(USB4PCIeTunnel));
    if (!tunnel) {
        return USB4_ERR_NO_MEMORY;
    }

    /* 어댑터 찾기 */
    tunnel->upstream = usb4_find_pcie_adapter(&ctx->host_router, true);
    tunnel->downstream = usb4_find_pcie_adapter(router, false);

    if (!tunnel->upstream || !tunnel->downstream) {
        USB4_ERR("Failed to find PCIe adapters");
        free(tunnel->upstream);
        free(tunnel->downstream);
        free(tunnel);
        return USB4_ERR_NOT_FOUND;
    }

    /* 경로 구성 */
    int ret = usb4_configure_path(tunnel);
    if (ret != USB4_SUCCESS) {
        USB4_ERR("Failed to configure tunnel path");
        free(tunnel->upstream);
        free(tunnel->downstream);
        free(tunnel);
        return ret;
    }

    /* 터널 활성화 */
    ret = usb4_activate_tunnel(tunnel);
    if (ret != USB4_SUCCESS) {
        USB4_ERR("Failed to activate tunnel");
        free(tunnel->upstream);
        free(tunnel->downstream);
        free(tunnel);
        return ret;
    }

    /* 컨텍스트에 등록 */
    ctx->active_tunnels[ctx->tunnel_count++] = tunnel;
    *tunnel_out = tunnel;

    USB4_LOG("PCIe tunnel created successfully (width=x%d, speed=Gen%d)",
             tunnel->pcie_link_width, tunnel->pcie_link_speed);

    return USB4_SUCCESS;
}

/**
 * 터널 경로 구성
 */
static int usb4_configure_path(USB4PCIeTunnel *tunnel)
{
    tunnel->state = USB4_TUNNEL_STATE_ACTIVATING;
    tunnel->path_length = 1;  /* 직접 연결 가정 */

    /* USB4는 최대 PCIe 3.0 x4 수준의 대역폭 지원 */
    tunnel->pcie_link_width = 4;
    tunnel->pcie_link_speed = 3;  /* Gen 3 */

    return USB4_SUCCESS;
}

/**
 * 터널 활성화
 */
static int usb4_activate_tunnel(USB4PCIeTunnel *tunnel)
{
    /* 어댑터 활성화 */
    tunnel->upstream->is_active = true;
    tunnel->downstream->is_active = true;

    /* 대역폭 할당 */
    tunnel->allocated_bw = USB4_PCIE_TUNNEL_BW_GBPS * 1000;  /* 32 Gbps */

    tunnel->state = USB4_TUNNEL_STATE_ACTIVE;

    return USB4_SUCCESS;
}

/**
 * PCIe 터널 제거
 */
int usb4_destroy_pcie_tunnel(USB4ControllerContext *ctx, USB4PCIeTunnel *tunnel)
{
    if (!ctx || !tunnel) {
        return USB4_ERR_INVALID_PARAM;
    }

    USB4_LOG("Destroying PCIe tunnel...");

    tunnel->state = USB4_TUNNEL_STATE_DEACTIVATING;

    /* 어댑터 비활성화 */
    if (tunnel->upstream) {
        tunnel->upstream->is_active = false;
        free(tunnel->upstream);
    }
    if (tunnel->downstream) {
        tunnel->downstream->is_active = false;
        free(tunnel->downstream);
    }

    /* 컨텍스트에서 제거 */
    for (int i = 0; i < ctx->tunnel_count; i++) {
        if (ctx->active_tunnels[i] == tunnel) {
            /* 배열 압축 */
            for (int j = i; j < ctx->tunnel_count - 1; j++) {
                ctx->active_tunnels[j] = ctx->active_tunnels[j + 1];
            }
            ctx->tunnel_count--;
            break;
        }
    }

    free(tunnel);

    USB4_LOG("PCIe tunnel destroyed");
    return USB4_SUCCESS;
}

/**
 * 라우터 레지스터 읽기
 */
int usb4_router_read_reg(USB4Router *router, uint32_t offset, uint32_t *value)
{
    if (!router || !value) {
        return USB4_ERR_INVALID_PARAM;
    }

#ifdef __APPLE__
    if (router->platform_data) {
        /* IOKit을 통한 레지스터 읽기 구현 */
        /* 실제 구현에서는 IOConnectCallScalarMethod 등 사용 */
    }
#endif

    /* 시뮬레이션용 기본값 */
    *value = 0;
    return USB4_SUCCESS;
}

/**
 * 라우터 레지스터 쓰기
 */
int usb4_router_write_reg(USB4Router *router, uint32_t offset, uint32_t value)
{
    if (!router) {
        return USB4_ERR_INVALID_PARAM;
    }

#ifdef __APPLE__
    if (router->platform_data) {
        /* IOKit을 통한 레지스터 쓰기 구현 */
    }
#endif

    return USB4_SUCCESS;
}

/**
 * 어댑터 레지스터 읽기
 */
int usb4_adapter_read_reg(USB4Adapter *adapter, uint32_t offset, uint32_t *value)
{
    if (!adapter || !value) {
        return USB4_ERR_INVALID_PARAM;
    }

    /* 라우터를 통해 접근 */
    uint32_t addr = (adapter->adapter_num << 8) | offset;
    return usb4_router_read_reg(adapter->router, addr, value);
}

/**
 * 어댑터 레지스터 쓰기
 */
int usb4_adapter_write_reg(USB4Adapter *adapter, uint32_t offset, uint32_t value)
{
    if (!adapter) {
        return USB4_ERR_INVALID_PARAM;
    }

    uint32_t addr = (adapter->adapter_num << 8) | offset;
    return usb4_router_write_reg(adapter->router, addr, value);
}

/**
 * 대역폭 할당 요청
 */
int usb4_request_bandwidth(USB4PCIeTunnel *tunnel,
                          uint32_t requested_bw,
                          uint32_t *allocated_bw)
{
    if (!tunnel || !allocated_bw) {
        return USB4_ERR_INVALID_PARAM;
    }

    /* USB4의 최대 PCIe 대역폭 */
    uint32_t max_bw = USB4_PCIE_TUNNEL_BW_GBPS * 1000;  /* 32 Gbps in Mbps */

    if (requested_bw > max_bw) {
        *allocated_bw = max_bw;
    } else {
        *allocated_bw = requested_bw;
    }

    tunnel->allocated_bw = *allocated_bw;

    USB4_LOG("Bandwidth allocated: %u Mbps (requested: %u Mbps)",
             *allocated_bw, requested_bw);

    return USB4_SUCCESS;
}

/**
 * 핫플러그 콜백 등록
 */
void usb4_register_hotplug_callback(USB4ControllerContext *ctx,
                                    void (*callback)(USB4Router*, bool))
{
    if (ctx) {
        ctx->hotplug_callback = callback;
    }
}

/**
 * 에러 콜백 등록
 */
void usb4_register_error_callback(USB4ControllerContext *ctx,
                                  void (*callback)(int, const char*))
{
    if (ctx) {
        ctx->error_callback = callback;
    }
}
