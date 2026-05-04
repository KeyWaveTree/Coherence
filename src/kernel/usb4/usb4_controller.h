/**
 * Coherence - USB4 Controller Interface
 *
 * Apple Silicon USB4 컨트롤러와의 저수준 통신을 담당합니다.
 * USB4는 Thunderbolt 3/4와 호환되며 PCIe 터널링을 지원합니다.
 */

#ifndef COHERENCE_USB4_CONTROLLER_H
#define COHERENCE_USB4_CONTROLLER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <IOKit/usb/IOUSBLib.h>
#endif

/* USB4 상수 정의 */
#define USB4_MAX_TUNNELS            8
#define USB4_MAX_ADAPTERS          64
#define USB4_PCIE_TUNNEL_BW_GBPS   32  /* USB4 Gen 3 기준 */

/* USB4 라우터 타입 */
typedef enum {
    USB4_ROUTER_TYPE_HOST = 0,
    USB4_ROUTER_TYPE_DEVICE,
    USB4_ROUTER_TYPE_HUB
} USB4RouterType;

/* USB4 어댑터 타입 */
typedef enum {
    USB4_ADAPTER_INACTIVE = 0,
    USB4_ADAPTER_LANE,
    USB4_ADAPTER_HOST_INTERFACE,
    USB4_ADAPTER_PCIE_DOWN,
    USB4_ADAPTER_PCIE_UP,
    USB4_ADAPTER_DP_IN,
    USB4_ADAPTER_DP_OUT,
    USB4_ADAPTER_USB3_DOWN,
    USB4_ADAPTER_USB3_UP
} USB4AdapterType;

/* USB4 터널 상태 */
typedef enum {
    USB4_TUNNEL_STATE_INACTIVE = 0,
    USB4_TUNNEL_STATE_ACTIVATING,
    USB4_TUNNEL_STATE_ACTIVE,
    USB4_TUNNEL_STATE_DEACTIVATING,
    USB4_TUNNEL_STATE_ERROR
} USB4TunnelState;

/* USB4 레지스터 오프셋 (Apple Silicon 기준) */
#define USB4_REG_ROUTER_CS0         0x00
#define USB4_REG_ROUTER_CS1         0x01
#define USB4_REG_ROUTER_CS2         0x02
#define USB4_REG_ROUTER_CS3         0x03
#define USB4_REG_ADAPTER_CS0        0x00
#define USB4_REG_ADAPTER_CS1        0x01
#define USB4_REG_ADAPTER_CS2        0x02
#define USB4_REG_ADP_PCIE_CS0       0x00
#define USB4_REG_ADP_PCIE_CS1       0x01

/* USB4 라우터 구조체 */
typedef struct USB4Router {
    uint64_t            route;              /* 라우터 경로 문자열 */
    USB4RouterType      type;               /* 라우터 타입 */
    uint8_t             adapter_count;      /* 어댑터 수 */
    uint32_t            vendor_id;          /* 벤더 ID */
    uint32_t            device_id;          /* 디바이스 ID */
    uint32_t            revision;           /* 리비전 */
    bool                is_connected;       /* 연결 상태 */
    void               *platform_data;      /* 플랫폼별 데이터 */
} USB4Router;

/* USB4 어댑터 구조체 */
typedef struct USB4Adapter {
    USB4Router         *router;             /* 부모 라우터 */
    uint8_t             adapter_num;        /* 어댑터 번호 */
    USB4AdapterType     type;               /* 어댑터 타입 */
    bool                is_active;          /* 활성화 상태 */
    uint32_t            max_bandwidth;      /* 최대 대역폭 (Mbps) */
} USB4Adapter;

/* USB4 PCIe 터널 구조체 */
typedef struct USB4PCIeTunnel {
    USB4Adapter        *upstream;           /* 업스트림 어댑터 */
    USB4Adapter        *downstream;         /* 다운스트림 어댑터 */
    USB4TunnelState     state;              /* 터널 상태 */
    uint32_t            allocated_bw;       /* 할당된 대역폭 */
    uint8_t             path_length;        /* 경로 길이 */

    /* PCIe 구성 */
    uint8_t             pcie_link_width;    /* PCIe 링크 폭 */
    uint8_t             pcie_link_speed;    /* PCIe 링크 속도 */
} USB4PCIeTunnel;

/* USB4 컨트롤러 컨텍스트 */
typedef struct USB4ControllerContext {
    USB4Router          host_router;        /* 호스트 라우터 */
    USB4Router         *connected_routers[USB4_MAX_TUNNELS];
    USB4PCIeTunnel     *active_tunnels[USB4_MAX_TUNNELS];
    uint8_t             router_count;       /* 연결된 라우터 수 */
    uint8_t             tunnel_count;       /* 활성 터널 수 */

    /* 플랫폼별 핸들 */
#ifdef __APPLE__
    io_service_t        service;
    io_connect_t        connection;
#endif

    /* 콜백 */
    void              (*hotplug_callback)(USB4Router *router, bool connected);
    void              (*error_callback)(int error_code, const char *message);
} USB4ControllerContext;

/* 함수 선언 */

/**
 * USB4 컨트롤러 초기화
 * @param ctx 컨트롤러 컨텍스트
 * @return 0 성공, 음수 실패
 */
int usb4_controller_init(USB4ControllerContext *ctx);

/**
 * USB4 컨트롤러 종료
 * @param ctx 컨트롤러 컨텍스트
 */
void usb4_controller_shutdown(USB4ControllerContext *ctx);

/**
 * 연결된 라우터 스캔
 * @param ctx 컨트롤러 컨텍스트
 * @return 발견된 라우터 수
 */
int usb4_scan_routers(USB4ControllerContext *ctx);

/**
 * PCIe 터널 생성
 * @param ctx 컨트롤러 컨텍스트
 * @param router 대상 라우터
 * @param tunnel 터널 구조체 출력
 * @return 0 성공, 음수 실패
 */
int usb4_create_pcie_tunnel(USB4ControllerContext *ctx,
                            USB4Router *router,
                            USB4PCIeTunnel **tunnel);

/**
 * PCIe 터널 제거
 * @param ctx 컨트롤러 컨텍스트
 * @param tunnel 제거할 터널
 * @return 0 성공, 음수 실패
 */
int usb4_destroy_pcie_tunnel(USB4ControllerContext *ctx,
                             USB4PCIeTunnel *tunnel);

/**
 * 라우터 레지스터 읽기
 * @param router 라우터
 * @param offset 레지스터 오프셋
 * @param value 값 출력
 * @return 0 성공, 음수 실패
 */
int usb4_router_read_reg(USB4Router *router, uint32_t offset, uint32_t *value);

/**
 * 라우터 레지스터 쓰기
 * @param router 라우터
 * @param offset 레지스터 오프셋
 * @param value 쓸 값
 * @return 0 성공, 음수 실패
 */
int usb4_router_write_reg(USB4Router *router, uint32_t offset, uint32_t value);

/**
 * 어댑터 레지스터 읽기
 * @param adapter 어댑터
 * @param offset 레지스터 오프셋
 * @param value 값 출력
 * @return 0 성공, 음수 실패
 */
int usb4_adapter_read_reg(USB4Adapter *adapter, uint32_t offset, uint32_t *value);

/**
 * 어댑터 레지스터 쓰기
 * @param adapter 어댑터
 * @param offset 레지스터 오프셋
 * @param value 쓸 값
 * @return 0 성공, 음수 실패
 */
int usb4_adapter_write_reg(USB4Adapter *adapter, uint32_t offset, uint32_t value);

/**
 * 대역폭 할당 요청
 * @param tunnel 터널
 * @param requested_bw 요청 대역폭 (Mbps)
 * @param allocated_bw 할당된 대역폭 출력 (Mbps)
 * @return 0 성공, 음수 실패
 */
int usb4_request_bandwidth(USB4PCIeTunnel *tunnel,
                          uint32_t requested_bw,
                          uint32_t *allocated_bw);

/**
 * 핫플러그 콜백 등록
 * @param ctx 컨트롤러 컨텍스트
 * @param callback 콜백 함수
 */
void usb4_register_hotplug_callback(USB4ControllerContext *ctx,
                                    void (*callback)(USB4Router*, bool));

/**
 * 에러 콜백 등록
 * @param ctx 컨트롤러 컨텍스트
 * @param callback 콜백 함수
 */
void usb4_register_error_callback(USB4ControllerContext *ctx,
                                  void (*callback)(int, const char*));

#endif /* COHERENCE_USB4_CONTROLLER_H */
