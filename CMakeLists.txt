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

# PORTABLE BUILD. Backends: CPU (portable OpenMP — Windows / Linux / macOS)
# and optional CUDA (NVIDIA GPUs; sm_75 Tesla T4 is the reference target,
# e.g. Kaggle). The CPU backend always builds, so the engine runs everywhere
# even without a GPU toolkit. Metal / Vulkan / TPU have no native backend:
# --device metal|vulkan falls back to CPU with a warning (see tools/main.cpp).
# ---------------------------------------------------------------- options
option(GAI_ENABLE_CUDA   "Build CUDA backend if a toolkit is found" ON)
option(GAI_ENABLE_OPENMP "Use OpenMP for CPU kernels"               ON)
option(GAI_BUILD_TESTS   "Build unit / gradient / benchmark tests"  ON)
option(GAI_NATIVE_ARCH   "Compile with -march=native"               OFF)
option(GAI_ENABLE_NCCL   "Build NCCL multi-GPU support"             ON)
option(GAI_ENABLE_PARQUET "Build native Parquet input via Apache Arrow" OFF)

# ---------------------------------------------------------------- compiler flags
# Flags are applied ONLY to C++ targets (via gai_apply_cxx_flags), never to the
# CUDA target. nvcc rejects flags like -march=native / -ffast-math, which is the
# classic "many errors" failure when the GPU build runs on Kaggle.
function(gai_apply_cxx_flags target)
    if(MSVC)
        # /MP = parallel compilation inside one project (fastest local builds).
        # /WX = zero-warning policy: any warning fails the build (no silent rot).
        target_compile_options(${target} PRIVATE /W3 /WX /wd5105 /permissive- /Zc:preprocessor /bigobj /utf-8 /MP)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        # -Werror = zero-warning policy on GCC/Clang (Kaggle). Unused-param kept
        # off: dispatch wrappers intentionally name all params for symmetry.
        # NOTE: -ffast-math is removed because -ffinite-math-only breaks std::isfinite/isnan
        # which is required for dynamic loss scaling to detect FP16 overflow.
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter -Wno-missing-field-initializers)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
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
            # Default: sm_75 keeps nvcc fast on Kaggle T4 (one arch, no fatbins).
            # Other GPUs: rebuild with -DCMAKE_CUDA_ARCHITECTURES=<sm> (e.g. 86
            # for RTX 3090, 89 for 4090, 80 for A100, 120 for RTX 5060/Blackwell)
            # or NATIVE (CMake >= 3.24). setup.sh detects this automatically.
            # sm_120 needs CUDA toolkit >= 12.8 (setup.sh gates this loudly).
            # Kernels are arch-generic CUDA (no sm_75 PTX/intrinsics): Blackwell
            # builds run correctly; launch tuning is T4-shaped (retune with
            # Nsight on the target GPU for peak occupancy).
            # No CUDA toolkit at all -> portable CPU-only build (runs anywhere).
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

# ---------------------------------------------------------------- Parquet (THE lake input, Hermes English)
# Native .parquet reading needs Apache Arrow C++ (large dependency, absent on
# most machines). OFF by default for local CPU builds; Kaggle training MUST
# use kaggle/setup.sh --with-parquet which installs Arrow and turns this ON.
# There is NO JSON file route anymore (parquet-only project): the supplied
# english_parquet lake is read directly into .gbin shards.
set(GAI_HAVE_PARQUET OFF)
if(GAI_ENABLE_PARQUET)
    find_package(Arrow QUIET)
    find_package(Parquet QUIET)
    if(Arrow_FOUND AND Parquet_FOUND)
        set(GAI_HAVE_PARQUET ON)
        target_link_libraries(gai_core PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
        target_compile_definitions(gai_core PUBLIC GAI_PARQUET=1)
        message(STATUS "[ghassan-ai] parquet: Arrow ${Arrow_VERSION} (native lake input enabled)")
    else()
        message(WARNING "[ghassan-ai] parquet: GAI_ENABLE_PARQUET=ON but Arrow/Parquet not found -> parquet route disabled (install Arrow: kaggle/setup.sh --with-parquet)")
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

# ---------------------------------------------------------------- gemini-chat (optional, C++ + libcurl)
# Maybni ghir ila CURL MSVC-compatible kayna (vcpkg install curl + toolchain).
# MinGW/msys curl (msys64/ucrt64) makhdamch m3a MSVC linker/headers -> OFF automatically.
# Ila OFF, khdem b tools/gemini_chat.py (stdlib only, bla install, nefs log format).
option(GAI_BUILD_GEMINI_CHAT "Build C++ gemini-chat (needs MSVC-compatible libcurl)" ON)
find_package(CURL QUIET)
set(GAI_HAVE_USABLE_CURL OFF)
if(CURL_FOUND AND GAI_BUILD_GEMINI_CHAT)
    if(MSVC AND "${CURL_INCLUDE_DIR}" MATCHES "[Mm]ingw|[Mm]sys|ucrt64|[Cc]ygwin")
        message(STATUS "[ghassan-ai] gemini-chat : OFF (found CURL is MinGW/msys: ${CURL_INCLUDE_DIR}; use tools/gemini_chat.py or vcpkg curl)")
    else()
        set(GAI_HAVE_USABLE_CURL ON)
    endif()
endif()
if(GAI_HAVE_USABLE_CURL AND EXISTS "${CMAKE_SOURCE_DIR}/tools/gemini_chat.cpp")
    add_executable(gemini-chat tools/gemini_chat.cpp)
    target_link_libraries(gemini-chat PRIVATE CURL::libcurl)
    gai_apply_cxx_flags(gemini-chat)
    message(STATUS "[ghassan-ai] gemini-chat : ON (libcurl)")
elseif(GAI_BUILD_GEMINI_CHAT AND NOT (MSVC AND CURL_FOUND AND "${CURL_INCLUDE_DIR}" MATCHES "[Mm]ingw|[Mm]sys|ucrt64|[Cc]ygwin"))
    message(STATUS "[ghassan-ai] gemini-chat : OFF (no usable libcurl or tools/gemini_chat.cpp missing; use tools/gemini_chat.py)")
endif()

# ---------------------------------------------------------------- tests (portable: cpu + cuda)
# NOTE: test_gradcheck.cpp is picked up by the glob below (gradient safety net).
if(GAI_BUILD_TESTS)
    enable_testing()
    file(GLOB GAI_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/test_*.cpp)
    foreach(tsrc ${GAI_TEST_SOURCES})
        get_filename_component(tname ${tsrc} NAME_WE)
        add_executable(${tname} ${tsrc} ${CMAKE_SOURCE_DIR}/tools/cli_common.cpp)
        target_link_libraries(${tname} PRIVATE gai_core)
        gai_apply_cxx_flags(${tname})
        add_test(NAME ${tname} COMMAND ${tname} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        # FIX P2 (hang blocks Kaggle ctest): no TIMEOUT meant a hung test
        # stalled the whole cell forever. 120s is generous for CPU unit tests.
        set_tests_properties(${tname} PROPERTIES TIMEOUT 120)
    endforeach()
endif()

message(STATUS "[ghassan-ai] build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "[ghassan-ai] cuda       : ${GAI_HAVE_CUDA} (ref sm_75 T4; override with -DCMAKE_CUDA_ARCHITECTURES)")
message(STATUS "[ghassan-ai] nccl       : ${GAI_HAVE_NCCL}")
message(STATUS "[ghassan-ai] parquet    : ${GAI_HAVE_PARQUET} (THE lake input; no JSON fallback)")
