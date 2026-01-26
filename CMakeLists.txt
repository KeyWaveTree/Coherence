# CudaBridge - CMake Build Configuration
#
# Apple Silicon Mac에서 eGPU를 통한 CUDA 연산 지원

cmake_minimum_required(VERSION 3.20)
project(CudaBridge VERSION 1.0.0 LANGUAGES C CXX)

# C/C++ 표준
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 옵션
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_DEBUG "Enable debug output" OFF)
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

# macOS 확인
if(NOT APPLE)
    message(WARNING "CudaBridge is designed for macOS with Apple Silicon")
endif()

# Apple Silicon 확인
if(APPLE)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT ARCH STREQUAL "arm64")
        message(WARNING "CudaBridge requires Apple Silicon (arm64)")
    endif()
endif()

# 컴파일 플래그
add_compile_options(-Wall -Wextra -Wpedantic)

if(ENABLE_DEBUG)
    add_compile_definitions(DEBUG)
    add_compile_options(-g -O0)
else()
    add_compile_options(-O2)
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

# macOS 프레임워크
if(APPLE)
    find_library(IOKIT_FRAMEWORK IOKit)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    find_library(SECURITY_FRAMEWORK Security)
endif()

# 소스 파일 수집
set(KERNEL_SOURCES
    src/kernel/usb4/usb4_controller.c
    src/kernel/pcie/pcie_tunnel.c
)

set(DRIVER_SOURCES
    src/driver/nvidia/nvidia_gpu.c
)

set(CUDA_SOURCES
    src/cuda/runtime/cuda_runtime.c
    src/cuda/memory/memory_manager.c
)

set(USERSPACE_SOURCES
    src/userspace/lib/cudabridge.c
)

# 메인 라이브러리
add_library(cudabridge
    ${KERNEL_SOURCES}
    ${DRIVER_SOURCES}
    ${CUDA_SOURCES}
    ${USERSPACE_SOURCES}
)

target_include_directories(cudabridge PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/userspace/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(cudabridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# macOS 프레임워크 링크
if(APPLE)
    target_link_libraries(cudabridge PRIVATE
        ${IOKIT_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
    )
endif()

# pthread
find_package(Threads REQUIRED)
target_link_libraries(cudabridge PRIVATE Threads::Threads)

# 라이브러리 속성
set_target_properties(cudabridge PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME cudabridge
)

# 정적 라이브러리도 빌드
add_library(cudabridge_static STATIC
    ${KERNEL_SOURCES}
    ${DRIVER_SOURCES}
    ${CUDA_SOURCES}
    ${USERSPACE_SOURCES}
)

target_include_directories(cudabridge_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/userspace/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(cudabridge_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(APPLE)
    target_link_libraries(cudabridge_static PRIVATE
        ${IOKIT_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
        ${SECURITY_FRAMEWORK}
    )
endif()

target_link_libraries(cudabridge_static PRIVATE Threads::Threads)

set_target_properties(cudabridge_static PROPERTIES
    OUTPUT_NAME cudabridge
)

# 테스트 빌드
if(BUILD_TESTS)
    enable_testing()

    add_executable(test_basic tests/test_basic.c)
    target_link_libraries(test_basic cudabridge)
    add_test(NAME basic_test COMMAND test_basic)

    add_executable(test_memory tests/test_memory.c)
    target_link_libraries(test_memory cudabridge)
    add_test(NAME memory_test COMMAND test_memory)

    add_executable(test_stream tests/test_stream.c)
    target_link_libraries(test_stream cudabridge)
    add_test(NAME stream_test COMMAND test_stream)

    # ARM 호환성 테스트 (라이브러리 의존성 없음)
    add_executable(test_arm_compat tests/test_arm_compat.c)
    add_test(NAME arm_compat_test COMMAND test_arm_compat)
endif()

# 예제 빌드
if(BUILD_EXAMPLES)
    add_executable(example_vector_add src/userspace/examples/vector_add.c)
    target_link_libraries(example_vector_add cudabridge)

    add_executable(example_device_info src/userspace/examples/device_info.c)
    target_link_libraries(example_device_info cudabridge)

    add_executable(example_bandwidth src/userspace/examples/bandwidth_test.c)
    target_link_libraries(example_bandwidth cudabridge)
endif()

# 설치 규칙
include(GNUInstallDirs)

install(TARGETS cudabridge cudabridge_static
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/userspace/include/cudabridge.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# pkg-config 파일 생성
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cudabridge.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/cudabridge.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cudabridge.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# CMake 패키지 설정
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/CudaBridgeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CudaBridgeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaBridge
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CudaBridgeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CudaBridgeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CudaBridgeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaBridge
)

# 요약 출력
message(STATUS "")
message(STATUS "CudaBridge ${PROJECT_VERSION} Configuration:")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Debug output: ${ENABLE_DEBUG}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
