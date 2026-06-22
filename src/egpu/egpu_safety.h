/**
 * CudaBridge - eGPU Connection Safety Manager
 *
 * Thunderbolt/USB4를 통해 연결된 eGPU의 안전한 연결, 해제, 오류 복구를
 * 관리합니다. 핫플러그 감지, 연결 상태 모니터링, 자동 복구 메커니즘을
 * 제공합니다.
 */

#ifndef CUDABRIDGE_EGPU_SAFETY_H
#define CUDABRIDGE_EGPU_SAFETY_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* eGPU 연결 상태 */
typedef enum {
    EGPU_STATE_DISCONNECTED = 0,    /* 연결 안 됨 */
    EGPU_STATE_DETECTING,           /* 감지 중 */
    EGPU_STATE_AUTHENTICATING,      /* 인증 중 (TB 보안) */
    EGPU_STATE_TUNNEL_SETUP,        /* PCIe 터널 설정 중 */
    EGPU_STATE_INITIALIZING,        /* GPU 초기화 중 */
    EGPU_STATE_CONNECTED,           /* 정상 연결 */
    EGPU_STATE_DEGRADED,            /* 성능 저하 상태 */
    EGPU_STATE_ERROR,               /* 오류 발생 */
    EGPU_STATE_RECOVERING,          /* 복구 중 */
    EGPU_STATE_SAFE_DISCONNECT,     /* 안전 해제 진행 중 */
    EGPU_STATE_FORCE_DISCONNECT     /* 강제 해제 진행 중 */
} EGPUConnectionState;

/* eGPU 오류 코드 */
typedef enum {
    EGPU_ERR_NONE = 0,
    EGPU_ERR_DEVICE_NOT_FOUND,          /* 장치를 찾을 수 없음 */
    EGPU_ERR_INCOMPATIBLE_DEVICE,       /* 호환 불가 장치 */
    EGPU_ERR_AUTH_FAILED,               /* Thunderbolt 인증 실패 */
    EGPU_ERR_TUNNEL_FAILED,             /* PCIe 터널 생성 실패 */
    EGPU_ERR_BANDWIDTH_LOW,             /* 대역폭 부족 */
    EGPU_ERR_POWER_INSUFFICIENT,        /* 전력 부족 */
    EGPU_ERR_LINK_UNSTABLE,             /* 링크 불안정 */
    EGPU_ERR_TIMEOUT,                   /* 연결 타임아웃 */
    EGPU_ERR_GPU_INIT_FAILED,           /* GPU 초기화 실패 */
    EGPU_ERR_DATA_CORRUPTION,           /* 데이터 전송 손상 */
    EGPU_ERR_UNEXPECTED_DISCONNECT,     /* 예기치 않은 연결 끊김 */
    EGPU_ERR_THERMAL_SHUTDOWN,          /* 과열로 인한 중단 */
    EGPU_ERR_DRIVER_MISMATCH,           /* 드라이버 버전 불일치 */
    EGPU_ERR_RECOVERY_FAILED            /* 복구 실패 */
} EGPUError;

/* 연결 유형 */
typedef enum {
    EGPU_CONN_USB4 = 0,
    EGPU_CONN_THUNDERBOLT3,
    EGPU_CONN_THUNDERBOLT4,
    EGPU_CONN_UNKNOWN
} EGPUConnectionType;

/* 건강 상태 지표 */
typedef struct EGPUHealthMetrics {
    float           link_error_rate;    /* 링크 오류율 (0.0-1.0) */
    float           bandwidth_util;     /* 대역폭 사용률 (0.0-1.0) */
    int             temperature;        /* GPU 온도 (°C) */
    int             power_draw;         /* 소비 전력 (W) */
    uint64_t        packets_sent;       /* 전송 패킷 수 */
    uint64_t        packets_received;   /* 수신 패킷 수 */
    uint64_t        errors_corrected;   /* 보정된 오류 수 */
    uint64_t        errors_uncorrected; /* 보정 불가 오류 수 */
    uint32_t        retransmissions;    /* 재전송 횟수 */
    struct timespec last_heartbeat;     /* 마지막 정상 통신 시간 */
} EGPUHealthMetrics;

/* 안전 정책 설정 */
typedef struct EGPUSafetyPolicy {
    int             max_retry_count;        /* 최대 재시도 횟수 (기본: 3) */
    int             retry_delay_ms;         /* 재시도 간격 (ms, 기본: 500) */
    int             heartbeat_interval_ms;  /* 하트비트 간격 (ms, 기본: 1000) */
    int             heartbeat_timeout_ms;   /* 하트비트 타임아웃 (ms, 기본: 5000) */
    float           error_rate_threshold;   /* 오류율 임계값 (기본: 0.01) */
    int             thermal_limit;          /* 온도 제한 (°C, 기본: 90) */
    int             enable_auto_recovery;   /* 자동 복구 활성화 (기본: 1) */
    int             enable_data_integrity;  /* 데이터 무결성 검증 (기본: 1) */
    int             force_disconnect_on_incompatible;  /* 호환불가시 강제해제 (기본: 1) */
    int             safe_disconnect_timeout_ms;  /* 안전 해제 타임아웃 (기본: 10000) */
} EGPUSafetyPolicy;

/* eGPU 장치 정보 */
typedef struct EGPUDeviceInfo {
    uint16_t        vendor_id;
    uint16_t        device_id;
    char            name[256];
    char            serial[64];
    EGPUConnectionType conn_type;
    uint32_t        max_bandwidth_mbps;     /* 최대 대역폭 (Mbps) */
    int             power_budget_watts;     /* 전력 예산 (W) */
    int             is_compatible;          /* 호환 여부 */
    char            incompatible_reason[256]; /* 비호환 사유 */
} EGPUDeviceInfo;

/* 이벤트 콜백 */
typedef void (*EGPUEventCallback)(EGPUConnectionState state,
                                   EGPUError error,
                                   const char *message,
                                   void *user_data);

/* ========== 연결 관리 ========== */

/**
 * eGPU 안전 관리자 초기화
 */
int egpu_safety_init(const EGPUSafetyPolicy *policy);

/**
 * eGPU 안전 관리자 종료
 */
void egpu_safety_shutdown(void);

/**
 * 기본 안전 정책 가져오기
 */
EGPUSafetyPolicy egpu_safety_default_policy(void);

/**
 * eGPU 연결 시도
 * 장치 감지 → 호환성 검사 → 인증 → 터널 설정 → 초기화 순서로 진행
 *
 * @param device_index 장치 인덱스 (-1이면 자동 감지)
 * @return EGPU_ERR_NONE 성공, 그 외 에러 코드
 */
EGPUError egpu_connect(int device_index);

/**
 * eGPU 안전 해제
 * 진행 중인 작업 완료 대기 → 메모리 해제 → 터널 해제 순서로 진행
 *
 * @return EGPU_ERR_NONE 성공
 */
EGPUError egpu_safe_disconnect(void);

/**
 * eGPU 강제 해제
 * 긴급 상황에서 즉시 연결 해제
 *
 * @param reason 강제 해제 사유
 * @return EGPU_ERR_NONE 성공
 */
EGPUError egpu_force_disconnect(const char *reason);

/* ========== 상태 모니터링 ========== */

/**
 * 현재 연결 상태 조회
 */
EGPUConnectionState egpu_get_state(void);

/**
 * 현재 오류 코드 조회
 */
EGPUError egpu_get_last_error(void);

/**
 * 오류 메시지 문자열 반환
 */
const char* egpu_error_string(EGPUError error);

/**
 * 건강 상태 지표 조회
 */
int egpu_get_health(EGPUHealthMetrics *metrics);

/**
 * 장치 정보 조회
 */
int egpu_get_device_info(EGPUDeviceInfo *info);

/* ========== 복구 메커니즘 ========== */

/**
 * 수동 복구 시도
 */
EGPUError egpu_try_recovery(void);

/**
 * 링크 재설정 (소프트 리셋)
 */
EGPUError egpu_reset_link(void);

/**
 * 전체 리셋 (하드 리셋)
 */
EGPUError egpu_hard_reset(void);

/* ========== 이벤트 및 콜백 ========== */

/**
 * 이벤트 콜백 등록
 */
void egpu_set_event_callback(EGPUEventCallback callback, void *user_data);

/**
 * 하트비트 모니터링 시작
 */
int egpu_start_monitoring(void);

/**
 * 하트비트 모니터링 중지
 */
void egpu_stop_monitoring(void);

/* ========== 호환성 검사 ========== */

/**
 * 장치 호환성 검사
 */
int egpu_check_compatibility(const EGPUDeviceInfo *info);

/**
 * 시스템 요구사항 검증
 */
int egpu_verify_system_requirements(void);

/* ========== 데이터 무결성 ========== */

/**
 * 전송 데이터 무결성 검증 (CRC32 기반)
 */
int egpu_verify_data_integrity(const void *data, size_t size,
                                uint32_t expected_crc);

/**
 * CRC32 계산
 */
uint32_t egpu_compute_crc32(const void *data, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* CUDABRIDGE_EGPU_SAFETY_H */
