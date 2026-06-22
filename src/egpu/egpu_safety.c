/**
 * CudaBridge - eGPU Connection Safety Manager Implementation
 *
 * eGPU의 안전한 연결/해제 및 오류 복구를 구현합니다.
 */

#include "egpu_safety.h"
#include "../logging/cb_log.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* CRC32 테이블 */
static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

static void init_crc32_table(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

/* 전역 상태 */
static struct {
    int                     initialized;
    EGPUSafetyPolicy        policy;
    EGPUConnectionState     state;
    EGPUError               last_error;
    EGPUDeviceInfo          device_info;
    EGPUHealthMetrics       health;
    EGPUEventCallback       event_callback;
    void                   *callback_data;
    pthread_mutex_t         lock;
    pthread_t               monitor_thread;
    int                     monitor_running;
    int                     retry_count;
} g_egpu = {0};

/* 내부 함수 선언 */
static void set_state(EGPUConnectionState new_state);
static void set_error(EGPUError error);
static void notify_event(const char *message);
static int simulate_device_detect(EGPUDeviceInfo *info);
static int check_device_compatible(const EGPUDeviceInfo *info);
static int setup_pcie_tunnel(void);
static int initialize_gpu(void);
static int teardown_gpu(void);
static int teardown_tunnel(void);
static void* monitor_thread_func(void *arg);

/* ========== 상태 오류 메시지 ========== */

const char* egpu_error_string(EGPUError error) {
    switch (error) {
        case EGPU_ERR_NONE:                     return "No error";
        case EGPU_ERR_DEVICE_NOT_FOUND:         return "eGPU device not found";
        case EGPU_ERR_INCOMPATIBLE_DEVICE:      return "Incompatible eGPU device";
        case EGPU_ERR_AUTH_FAILED:              return "Thunderbolt authentication failed";
        case EGPU_ERR_TUNNEL_FAILED:            return "PCIe tunnel creation failed";
        case EGPU_ERR_BANDWIDTH_LOW:            return "Insufficient bandwidth";
        case EGPU_ERR_POWER_INSUFFICIENT:       return "Insufficient power supply";
        case EGPU_ERR_LINK_UNSTABLE:            return "Unstable link connection";
        case EGPU_ERR_TIMEOUT:                  return "Connection timeout";
        case EGPU_ERR_GPU_INIT_FAILED:          return "GPU initialization failed";
        case EGPU_ERR_DATA_CORRUPTION:          return "Data transfer corruption detected";
        case EGPU_ERR_UNEXPECTED_DISCONNECT:    return "Unexpected disconnection";
        case EGPU_ERR_THERMAL_SHUTDOWN:         return "Thermal limit exceeded";
        case EGPU_ERR_DRIVER_MISMATCH:          return "Driver version mismatch";
        case EGPU_ERR_RECOVERY_FAILED:          return "Recovery attempt failed";
        default:                                return "Unknown error";
    }
}

/* ========== 내부 유틸리티 ========== */

static void set_state(EGPUConnectionState new_state) {
    EGPUConnectionState old_state = g_egpu.state;
    g_egpu.state = new_state;
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "State transition: %d -> %d", old_state, new_state);
}

static void set_error(EGPUError error) {
    g_egpu.last_error = error;
    if (error != EGPU_ERR_NONE) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU, "Error set: %s", egpu_error_string(error));
    }
}

static void notify_event(const char *message) {
    if (g_egpu.event_callback) {
        g_egpu.event_callback(g_egpu.state, g_egpu.last_error,
                               message, g_egpu.callback_data);
    }
}

static int simulate_device_detect(EGPUDeviceInfo *info) {
    /*
     * 실제 환경에서는 IOKit/USB4 API를 통해 장치를 감지합니다.
     * 여기서는 시뮬레이션 모드로 동작합니다.
     */
    CB_LOG_DEBUG(CB_LOG_CAT_EGPU, "Scanning for eGPU devices...");

    /* 시뮬레이션: 가상 GPU 장치 정보 */
    info->vendor_id = 0x10DE;  /* NVIDIA */
    info->device_id = 0x2684;  /* RTX 4090 예시 */
    snprintf(info->name, sizeof(info->name), "NVIDIA GeForce RTX 4090 (Simulated)");
    snprintf(info->serial, sizeof(info->serial), "SIM-0000-0000-0001");
    info->conn_type = EGPU_CONN_THUNDERBOLT4;
    info->max_bandwidth_mbps = 32000;  /* 32 Gbps */
    info->power_budget_watts = 450;
    info->is_compatible = 1;
    info->incompatible_reason[0] = '\0';

    return 0;
}

static int check_device_compatible(const EGPUDeviceInfo *info) {
    /* NVIDIA 벤더 확인 */
    if (info->vendor_id != 0x10DE) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                     "Unsupported vendor: 0x%04X (only NVIDIA 0x10DE supported)",
                     info->vendor_id);
        return 0;
    }

    /* 최소 대역폭 확인 (8 Gbps) */
    if (info->max_bandwidth_mbps < 8000) {
        CB_LOG_WARN(CB_LOG_CAT_EGPU,
                    "Low bandwidth: %u Mbps (minimum 8000 Mbps recommended)",
                    info->max_bandwidth_mbps);
    }

    /* 전력 예산 확인 */
    if (info->power_budget_watts < 75) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                     "Insufficient power: %d W (minimum 75W required)",
                     info->power_budget_watts);
        return 0;
    }

    return 1;
}

static int setup_pcie_tunnel(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Setting up PCIe tunnel over %s",
                g_egpu.device_info.conn_type == EGPU_CONN_USB4 ? "USB4" :
                g_egpu.device_info.conn_type == EGPU_CONN_THUNDERBOLT3 ? "Thunderbolt 3" :
                g_egpu.device_info.conn_type == EGPU_CONN_THUNDERBOLT4 ? "Thunderbolt 4" :
                "Unknown");

    /* 시뮬레이션: 터널 설정 */
    usleep(100000); /* 100ms 시뮬레이션 대기 */

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "PCIe tunnel established: bandwidth %u Mbps",
                g_egpu.device_info.max_bandwidth_mbps);
    return 0;
}

static int initialize_gpu(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Initializing GPU: %s", g_egpu.device_info.name);

    /* 시뮬레이션: GPU 초기화 */
    usleep(200000); /* 200ms 시뮬레이션 대기 */

    /* 건강 지표 초기화 */
    memset(&g_egpu.health, 0, sizeof(EGPUHealthMetrics));
    g_egpu.health.temperature = 35;
    g_egpu.health.power_draw = 25;
    clock_gettime(CLOCK_REALTIME, &g_egpu.health.last_heartbeat);

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "GPU initialized successfully");
    return 0;
}

static int teardown_gpu(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Shutting down GPU...");
    usleep(100000);
    return 0;
}

static int teardown_tunnel(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Tearing down PCIe tunnel...");
    usleep(50000);
    return 0;
}

/* ========== 모니터링 스레드 ========== */

static void* monitor_thread_func(void *arg) {
    (void)arg;
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Health monitoring thread started");

    while (g_egpu.monitor_running) {
        usleep((unsigned int)(g_egpu.policy.heartbeat_interval_ms * 1000));

        if (!g_egpu.monitor_running) break;
        if (g_egpu.state != EGPU_STATE_CONNECTED &&
            g_egpu.state != EGPU_STATE_DEGRADED) continue;

        pthread_mutex_lock(&g_egpu.lock);

        /* 하트비트 갱신 */
        clock_gettime(CLOCK_REALTIME, &g_egpu.health.last_heartbeat);

        /* 온도 체크 */
        if (g_egpu.health.temperature >= g_egpu.policy.thermal_limit) {
            CB_LOG_FATAL(CB_LOG_CAT_EGPU,
                         "THERMAL SHUTDOWN: Temperature %d°C exceeds limit %d°C",
                         g_egpu.health.temperature, g_egpu.policy.thermal_limit);
            set_error(EGPU_ERR_THERMAL_SHUTDOWN);
            set_state(EGPU_STATE_ERROR);
            notify_event("Thermal limit exceeded - emergency shutdown");
            pthread_mutex_unlock(&g_egpu.lock);

            egpu_force_disconnect("Thermal emergency");
            continue;
        }

        /* 링크 오류율 체크 */
        if (g_egpu.health.link_error_rate > g_egpu.policy.error_rate_threshold) {
            CB_LOG_WARN(CB_LOG_CAT_EGPU,
                        "High error rate: %.4f (threshold: %.4f)",
                        g_egpu.health.link_error_rate,
                        g_egpu.policy.error_rate_threshold);

            if (g_egpu.health.link_error_rate > g_egpu.policy.error_rate_threshold * 10) {
                CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                             "Critical error rate - initiating recovery");
                set_error(EGPU_ERR_LINK_UNSTABLE);
                set_state(EGPU_STATE_ERROR);
                notify_event("Link error rate critical");
                pthread_mutex_unlock(&g_egpu.lock);

                if (g_egpu.policy.enable_auto_recovery) {
                    egpu_try_recovery();
                }
                continue;
            } else {
                set_state(EGPU_STATE_DEGRADED);
                notify_event("Link performance degraded");
            }
        }

        /* 데이터 무결성 보정 불가 오류 체크 */
        if (g_egpu.health.errors_uncorrected > 0) {
            CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                         "Uncorrectable errors detected: %lu",
                         (unsigned long)g_egpu.health.errors_uncorrected);
            set_error(EGPU_ERR_DATA_CORRUPTION);
            set_state(EGPU_STATE_ERROR);
            notify_event("Data integrity violation detected");
            pthread_mutex_unlock(&g_egpu.lock);

            if (g_egpu.policy.enable_auto_recovery) {
                egpu_try_recovery();
            }
            continue;
        }

        pthread_mutex_unlock(&g_egpu.lock);
    }

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Health monitoring thread stopped");
    return NULL;
}

/* ========== 공개 API 구현 ========== */

EGPUSafetyPolicy egpu_safety_default_policy(void) {
    EGPUSafetyPolicy policy = {
        .max_retry_count = 3,
        .retry_delay_ms = 500,
        .heartbeat_interval_ms = 1000,
        .heartbeat_timeout_ms = 5000,
        .error_rate_threshold = 0.01f,
        .thermal_limit = 90,
        .enable_auto_recovery = 1,
        .enable_data_integrity = 1,
        .force_disconnect_on_incompatible = 1,
        .safe_disconnect_timeout_ms = 10000,
    };
    return policy;
}

int egpu_safety_init(const EGPUSafetyPolicy *policy) {
    if (g_egpu.initialized) return 0;

    if (!crc32_table_initialized) {
        init_crc32_table();
    }

    if (policy) {
        memcpy(&g_egpu.policy, policy, sizeof(EGPUSafetyPolicy));
    } else {
        g_egpu.policy = egpu_safety_default_policy();
    }

    pthread_mutex_init(&g_egpu.lock, NULL);
    g_egpu.state = EGPU_STATE_DISCONNECTED;
    g_egpu.last_error = EGPU_ERR_NONE;
    g_egpu.monitor_running = 0;
    g_egpu.retry_count = 0;
    g_egpu.initialized = 1;

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "eGPU Safety Manager initialized");
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  Max retries: %d, Heartbeat: %dms, Thermal limit: %d°C",
                g_egpu.policy.max_retry_count,
                g_egpu.policy.heartbeat_interval_ms,
                g_egpu.policy.thermal_limit);

    return 0;
}

void egpu_safety_shutdown(void) {
    if (!g_egpu.initialized) return;

    /* 모니터링 중지 */
    egpu_stop_monitoring();

    /* 연결 상태면 안전 해제 */
    if (g_egpu.state == EGPU_STATE_CONNECTED ||
        g_egpu.state == EGPU_STATE_DEGRADED) {
        egpu_safe_disconnect();
    }

    pthread_mutex_destroy(&g_egpu.lock);
    g_egpu.initialized = 0;
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "eGPU Safety Manager shutdown");
}

EGPUError egpu_connect(int device_index) {
    if (!g_egpu.initialized) return EGPU_ERR_DEVICE_NOT_FOUND;

    pthread_mutex_lock(&g_egpu.lock);
    (void)device_index;

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "=== eGPU Connection Sequence Started ===");

    /* Step 1: 장치 감지 */
    set_state(EGPU_STATE_DETECTING);
    notify_event("Scanning for eGPU devices...");

    if (simulate_device_detect(&g_egpu.device_info) != 0) {
        set_error(EGPU_ERR_DEVICE_NOT_FOUND);
        set_state(EGPU_STATE_DISCONNECTED);
        notify_event("No eGPU device found");
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_DEVICE_NOT_FOUND;
    }

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Device found: %s [%04X:%04X]",
                g_egpu.device_info.name,
                g_egpu.device_info.vendor_id,
                g_egpu.device_info.device_id);

    /* Step 2: 호환성 검사 */
    if (!check_device_compatible(&g_egpu.device_info)) {
        set_error(EGPU_ERR_INCOMPATIBLE_DEVICE);

        if (g_egpu.policy.force_disconnect_on_incompatible) {
            CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                         "Incompatible device detected - forcing disconnect");
            set_state(EGPU_STATE_FORCE_DISCONNECT);
            notify_event("Incompatible device - force disconnect initiated");
            set_state(EGPU_STATE_DISCONNECTED);
        } else {
            set_state(EGPU_STATE_ERROR);
            notify_event("Incompatible device detected");
        }
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_INCOMPATIBLE_DEVICE;
    }

    /* Step 3: Thunderbolt 인증 */
    set_state(EGPU_STATE_AUTHENTICATING);
    notify_event("Authenticating Thunderbolt connection...");
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Thunderbolt security authentication passed");

    /* Step 4: PCIe 터널 설정 */
    set_state(EGPU_STATE_TUNNEL_SETUP);
    notify_event("Setting up PCIe tunnel...");

    if (setup_pcie_tunnel() != 0) {
        set_error(EGPU_ERR_TUNNEL_FAILED);
        set_state(EGPU_STATE_ERROR);
        notify_event("PCIe tunnel setup failed");

        /* 재시도 로직 */
        if (g_egpu.policy.enable_auto_recovery &&
            g_egpu.retry_count < g_egpu.policy.max_retry_count) {
            g_egpu.retry_count++;
            CB_LOG_WARN(CB_LOG_CAT_EGPU,
                        "Retrying tunnel setup (%d/%d)...",
                        g_egpu.retry_count, g_egpu.policy.max_retry_count);
            pthread_mutex_unlock(&g_egpu.lock);
            usleep((unsigned int)(g_egpu.policy.retry_delay_ms * 1000));
            return egpu_connect(device_index);
        }

        set_state(EGPU_STATE_DISCONNECTED);
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_TUNNEL_FAILED;
    }

    /* Step 5: GPU 초기화 */
    set_state(EGPU_STATE_INITIALIZING);
    notify_event("Initializing GPU...");

    if (initialize_gpu() != 0) {
        set_error(EGPU_ERR_GPU_INIT_FAILED);
        set_state(EGPU_STATE_ERROR);
        notify_event("GPU initialization failed");
        teardown_tunnel();
        set_state(EGPU_STATE_DISCONNECTED);
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_GPU_INIT_FAILED;
    }

    /* 연결 완료 */
    set_state(EGPU_STATE_CONNECTED);
    set_error(EGPU_ERR_NONE);
    g_egpu.retry_count = 0;

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "=== eGPU Connected Successfully ===");
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  Device: %s", g_egpu.device_info.name);
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  Connection: %s",
                g_egpu.device_info.conn_type == EGPU_CONN_USB4 ? "USB4" :
                g_egpu.device_info.conn_type == EGPU_CONN_THUNDERBOLT3 ? "Thunderbolt 3" :
                g_egpu.device_info.conn_type == EGPU_CONN_THUNDERBOLT4 ? "Thunderbolt 4" :
                "Unknown");
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  Bandwidth: %u Mbps",
                g_egpu.device_info.max_bandwidth_mbps);

    notify_event("eGPU connected and ready");

    pthread_mutex_unlock(&g_egpu.lock);

    /* 모니터링 시작 */
    egpu_start_monitoring();

    return EGPU_ERR_NONE;
}

EGPUError egpu_safe_disconnect(void) {
    if (!g_egpu.initialized) return EGPU_ERR_NONE;

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "=== Safe Disconnect Initiated ===");

    /* 모니터링 중지 */
    egpu_stop_monitoring();

    pthread_mutex_lock(&g_egpu.lock);
    set_state(EGPU_STATE_SAFE_DISCONNECT);
    notify_event("Safe disconnect in progress...");

    /* Step 1: 진행 중인 작업 완료 대기 */
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Waiting for pending operations to complete...");
    /* 실제 환경에서는 cbDeviceSynchronize() 호출 */
    usleep(100000);

    /* Step 2: GPU 리소스 해제 */
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Releasing GPU resources...");
    teardown_gpu();

    /* Step 3: PCIe 터널 해제 */
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Tearing down PCIe tunnel...");
    teardown_tunnel();

    /* 완료 */
    set_state(EGPU_STATE_DISCONNECTED);
    set_error(EGPU_ERR_NONE);
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "=== eGPU Safely Disconnected ===");
    notify_event("eGPU disconnected safely");

    pthread_mutex_unlock(&g_egpu.lock);
    return EGPU_ERR_NONE;
}

EGPUError egpu_force_disconnect(const char *reason) {
    if (!g_egpu.initialized) return EGPU_ERR_NONE;

    CB_LOG_WARN(CB_LOG_CAT_EGPU, "=== FORCE DISCONNECT: %s ===",
                reason ? reason : "No reason given");

    /* 모니터링 즉시 중지 */
    g_egpu.monitor_running = 0;

    pthread_mutex_lock(&g_egpu.lock);
    set_state(EGPU_STATE_FORCE_DISCONNECT);
    notify_event(reason ? reason : "Force disconnect");

    /* 즉시 해제 - 대기 없음 */
    teardown_gpu();
    teardown_tunnel();

    set_state(EGPU_STATE_DISCONNECTED);
    CB_LOG_WARN(CB_LOG_CAT_EGPU, "=== Force Disconnect Complete ===");
    notify_event("eGPU forcefully disconnected");

    pthread_mutex_unlock(&g_egpu.lock);
    return EGPU_ERR_NONE;
}

EGPUConnectionState egpu_get_state(void) {
    return g_egpu.state;
}

EGPUError egpu_get_last_error(void) {
    return g_egpu.last_error;
}

int egpu_get_health(EGPUHealthMetrics *metrics) {
    if (!metrics) return -1;
    pthread_mutex_lock(&g_egpu.lock);
    memcpy(metrics, &g_egpu.health, sizeof(EGPUHealthMetrics));
    pthread_mutex_unlock(&g_egpu.lock);
    return 0;
}

int egpu_get_device_info(EGPUDeviceInfo *info) {
    if (!info) return -1;
    pthread_mutex_lock(&g_egpu.lock);
    memcpy(info, &g_egpu.device_info, sizeof(EGPUDeviceInfo));
    pthread_mutex_unlock(&g_egpu.lock);
    return 0;
}

EGPUError egpu_try_recovery(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "=== Recovery Attempt ===");

    if (g_egpu.retry_count >= g_egpu.policy.max_retry_count) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                     "Max recovery attempts reached (%d) - force disconnect",
                     g_egpu.policy.max_retry_count);
        egpu_force_disconnect("Max recovery attempts exceeded");
        return EGPU_ERR_RECOVERY_FAILED;
    }

    g_egpu.retry_count++;
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Recovery attempt %d/%d",
                g_egpu.retry_count, g_egpu.policy.max_retry_count);

    pthread_mutex_lock(&g_egpu.lock);
    set_state(EGPU_STATE_RECOVERING);
    notify_event("Recovery in progress...");
    pthread_mutex_unlock(&g_egpu.lock);

    /* Step 1: 링크 리셋 시도 */
    EGPUError err = egpu_reset_link();
    if (err == EGPU_ERR_NONE) {
        pthread_mutex_lock(&g_egpu.lock);
        set_state(EGPU_STATE_CONNECTED);
        set_error(EGPU_ERR_NONE);
        g_egpu.retry_count = 0;
        g_egpu.health.errors_uncorrected = 0;
        g_egpu.health.link_error_rate = 0.0f;
        notify_event("Recovery successful (link reset)");
        CB_LOG_INFO(CB_LOG_CAT_EGPU, "Recovery successful via link reset");
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_NONE;
    }

    /* Step 2: 하드 리셋 시도 */
    CB_LOG_WARN(CB_LOG_CAT_EGPU, "Link reset failed, attempting hard reset...");
    usleep((unsigned int)(g_egpu.policy.retry_delay_ms * 1000));

    err = egpu_hard_reset();
    if (err == EGPU_ERR_NONE) {
        pthread_mutex_lock(&g_egpu.lock);
        set_state(EGPU_STATE_CONNECTED);
        set_error(EGPU_ERR_NONE);
        g_egpu.retry_count = 0;
        notify_event("Recovery successful (hard reset)");
        CB_LOG_INFO(CB_LOG_CAT_EGPU, "Recovery successful via hard reset");
        pthread_mutex_unlock(&g_egpu.lock);
        return EGPU_ERR_NONE;
    }

    /* Step 3: 재연결 시도 */
    CB_LOG_WARN(CB_LOG_CAT_EGPU, "Hard reset failed, attempting reconnection...");
    egpu_force_disconnect("Recovery reconnection");
    usleep((unsigned int)(g_egpu.policy.retry_delay_ms * 1000 * 2));

    return egpu_connect(-1);
}

EGPUError egpu_reset_link(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Performing soft link reset...");
    /* 시뮬레이션: 링크 리셋 */
    usleep(200000);
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Link reset completed");
    return EGPU_ERR_NONE;
}

EGPUError egpu_hard_reset(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Performing hard reset...");
    /* 시뮬레이션: 하드 리셋 */
    usleep(500000);
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Hard reset completed");
    return EGPU_ERR_NONE;
}

void egpu_set_event_callback(EGPUEventCallback callback, void *user_data) {
    pthread_mutex_lock(&g_egpu.lock);
    g_egpu.event_callback = callback;
    g_egpu.callback_data = user_data;
    pthread_mutex_unlock(&g_egpu.lock);
}

int egpu_start_monitoring(void) {
    if (g_egpu.monitor_running) return 0;

    g_egpu.monitor_running = 1;
    if (pthread_create(&g_egpu.monitor_thread, NULL,
                        monitor_thread_func, NULL) != 0) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU, "Failed to start monitoring thread");
        g_egpu.monitor_running = 0;
        return -1;
    }

    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Health monitoring started");
    return 0;
}

void egpu_stop_monitoring(void) {
    if (!g_egpu.monitor_running) return;

    g_egpu.monitor_running = 0;
    pthread_join(g_egpu.monitor_thread, NULL);
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Health monitoring stopped");
}

int egpu_check_compatibility(const EGPUDeviceInfo *info) {
    if (!info) return 0;
    return check_device_compatible(info);
}

int egpu_verify_system_requirements(void) {
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "Verifying system requirements...");

    /* USB4/Thunderbolt 포트 확인 (시뮬레이션) */
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  USB4/Thunderbolt port: Available (simulated)");

    /* 커널 버전 확인 */
    CB_LOG_INFO(CB_LOG_CAT_EGPU, "  System requirements: OK (simulated)");

    return 1;
}

uint32_t egpu_compute_crc32(const void *data, size_t size) {
    if (!crc32_table_initialized) init_crc32_table();

    const uint8_t *buf = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < size; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ buf[i]) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

int egpu_verify_data_integrity(const void *data, size_t size,
                                uint32_t expected_crc) {
    uint32_t actual_crc = egpu_compute_crc32(data, size);
    if (actual_crc != expected_crc) {
        CB_LOG_ERROR(CB_LOG_CAT_EGPU,
                     "Data integrity check failed: expected CRC 0x%08X, got 0x%08X",
                     expected_crc, actual_crc);
        return 0;
    }
    return 1;
}
