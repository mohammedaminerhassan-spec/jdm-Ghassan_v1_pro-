cmake_minimum_required(VERSION 3.20)

project(ghassan_ai LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
    set(VcpkgApplocalDeps OFF CACHE BOOL "Disable vcpkg applocal dependencies copy" FORCE)
    set(CMAKE_VS_GLOBALS "VcpkgApplocalDeps=false")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# T4-ONLY BUILD. This project targets exactly two backends:
#   CPU (portable, OpenMP) and CUDA sm_75 (Tesla T4, e.g. Kaggle).
# Metal / Vulkan / TPU backends were removed on purpose: dead options slow
# down configuration, grow the build graph, and invite untested code paths.
# ---------------------------------------------------------------- options
option(GAI_ENABLE_CUDA   "Build CUDA backend if a toolkit is found" ON)
option(GAI_ENABLE_OPENMP "Use OpenMP for CPU kernels"               ON)
option(GAI_BUILD_TESTS   "Build unit / gradient / benchmark tests"  ON)
option(GAI_NATIVE_ARCH   "Compile with -march=native"               OFF)
option(GAI_ENABLE_NCCL   "Build NCCL multi-GPU support"             ON)

# ---------------------------------------------------------------- compiler flags
# Flags are applied ONLY to C++ targets (via gai_apply_cxx_flags), never to the
# CUDA target. nvcc rejects flags like -march=native / -ffast-math, which is the
# classic "many errors" failure when the GPU build runs on Kaggle.
function(gai_apply_cxx_flags target)
    if(MSVC)
        # /MP = parallel compilation inside one project (fastest local builds).
        # /WX = zero-warning policy: any warning fails the build (no silent rot).
        target_compile_options(${target} PRIVATE /W3 /WX /permissive- /Zc:preprocessor /bigobj /utf-8 /MP)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast>)
    else()
        # -Werror = zero-warning policy on GCC/Clang (Kaggle). Unused-param kept
        # off: dispatch wrappers intentionally name all params for symmetry.
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -Wno-missing-field-initializers)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-ffast-math>)
        if(GAI_NATIVE_ARCH)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------- OpenMP
if(GAI_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "[ghassan-ai] OpenMP: enabled")
    else()
        message(STATUS "[ghassan-ai] OpenMP: not found (single-threaded CPU kernels)")
    endif()
endif()

# ---------------------------------------------------------------- CUDA autodetect
set(GAI_HAVE_CUDA OFF)
if(GAI_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # T4-only: sm_75 keeps nvcc fast (one arch, no fatbins).
            # Override with -DCMAKE_CUDA_ARCHITECTURES=80 for A100, etc.
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(GAI_HAVE_CUDA ON)
        message(STATUS "[ghassan-ai] CUDA: ${CMAKE_CUDA_COMPILER_VERSION} arch=${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "[ghassan-ai] CUDA: no toolkit found -> CPU-only build")
    endif()
endif()

# ---------------------------------------------------------------- sources
file(GLOB_RECURSE GAI_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/core/*.cpp
     ${CMAKE_SOURCE_DIR}/tokenizer/*.cpp
     ${CMAKE_SOURCE_DIR}/model/*.cpp
     ${CMAKE_SOURCE_DIR}/format/*.cpp
     ${CMAKE_SOURCE_DIR}/training/*.cpp
     ${CMAKE_SOURCE_DIR}/dataset/*.cpp
     ${CMAKE_SOURCE_DIR}/inference/*.cpp
     ${CMAKE_SOURCE_DIR}/quantization/*.cpp
     ${CMAKE_SOURCE_DIR}/evaluation/*.cpp)

add_library(gai_core STATIC ${GAI_SOURCES})
target_include_directories(gai_core PUBLIC ${CMAKE_SOURCE_DIR})
gai_apply_cxx_flags(gai_core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gai_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gai_core PUBLIC GAI_OPENMP=1)
endif()

if(GAI_HAVE_CUDA)
    file(GLOB GAI_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cuda/*.cu)
    add_library(gai_cuda STATIC ${GAI_CUDA_SOURCES})
    target_include_directories(gai_cuda PUBLIC ${CMAKE_SOURCE_DIR})
    set_target_properties(gai_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(gai_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(gai_cuda PUBLIC GAI_CUDA=1)
    target_link_libraries(gai_cuda PUBLIC CUDA::cudart CUDA::cublas)
    target_link_libraries(gai_core PUBLIC gai_cuda)
    target_compile_definitions(gai_core PUBLIC GAI_CUDA=1)
endif()

# ---------------------------------------------------------------- NCCL (multi-GPU)
# No Findnccl module ships with CMake, so detect header + library directly.
set(GAI_HAVE_NCCL OFF)
if(GAI_HAVE_CUDA AND GAI_ENABLE_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include)
    find_library(NCCL_LIBRARY nccl
        HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        set(GAI_HAVE_NCCL ON)
        target_include_directories(gai_core PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(gai_core PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(gai_core PUBLIC GAI_NCCL=1)
        message(STATUS "[ghassan-ai] NCCL: ${NCCL_LIBRARY} (multi-GPU enabled)")
    else()
        message(STATUS "[ghassan-ai] NCCL: not found -> single-GPU only")
    endif()
endif()

if(WIN32)
    target_compile_definitions(gai_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

if(NOT MSVC)
    find_package(Threads REQUIRED)
    target_link_libraries(gai_core PUBLIC Threads::Threads)
endif()

# ---------------------------------------------------------------- executables
add_executable(ghassan-ai      tools/main.cpp tools/cli_common.cpp)
add_executable(train_tokenizer tools/train_tokenizer_main.cpp tools/cli_common.cpp)
add_executable(corpus_stats    tools/corpus_stats_main.cpp tools/cli_common.cpp)
add_executable(data_pipeline   tools/data_pipeline_main.cpp tools/cli_common.cpp)
add_executable(gai_train       tools/train_main.cpp tools/cli_common.cpp)

foreach(t ghassan-ai train_tokenizer corpus_stats data_pipeline gai_train)
    target_link_libraries(${t} PRIVATE gai_core)
    gai_apply_cxx_flags(${t})
endforeach()

# ---------------------------------------------------------------- tests (T4-only: cpu + cuda)
# NOTE: test_gradcheck.cpp is picked up by the glob below (gradient safety net).
if(GAI_BUILD_TESTS)
    enable_testing()
    file(GLOB GAI_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
    foreach(tsrc ${GAI_TEST_SOURCES})
        get_filename_component(tname ${tsrc} NAME_WE)
        add_executable(${tname} ${tsrc})
        target_link_libraries(${tname} PRIVATE gai_core)
        gai_apply_cxx_flags(${tname})
        add_test(NAME ${tname} COMMAND ${tname})
    endforeach()
endif()

message(STATUS "[ghassan-ai] build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "[ghassan-ai] cuda       : ${GAI_HAVE_CUDA} (T4-only, sm_75)")
message(STATUS "[ghassan-ai] nccl       : ${GAI_HAVE_NCCL}")
