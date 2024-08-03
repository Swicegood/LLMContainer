cmake_minimum_required(VERSION 3.18)  # Increased to 3.18 for better CUDA support
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
project(llava-server CUDA CXX)

# Set CMake policy CMP0104 to NEW
cmake_policy(SET CMP0104 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(LLAVA_CUDA "Enable CUDA support" OFF)
option(GGML_STATIC "ggml: static link libraries" ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

if (LLAVA_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DGGML_USE_CUBLAS)
    
    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)
endif()

# Add nlohmann/json
include_directories(./include)

# Force include all ggml source files
set(GGML_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-alloc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-backend.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-quants.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-aarch64.c
)

if(LLAVA_CUDA)
    list(APPEND GGML_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cuda.cu)
endif()

# [GGML options omitted for brevity]
# ... (keep the existing ggml options)
# general
option(GGML_STATIC "ggml: static link libraries"         OFF)
option(GGML_NATIVE "ggml: enable -march=native flag"     ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization" OFF)
option(GGML_CCACHE "ggml: use ccache if available"       ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(GGML_CPU_HBM     "ggml: use memkind for CPU HBM" OFF)

option(GGML_AVX         "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX2        "ggml: enable AVX2"             ${INS_ENB})
option(GGML_AVX512      "ggml: enable AVX512"           OFF)
option(GGML_AVX512_VBMI "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16 "ggml: enable AVX512-BF16"      OFF)
option(GGML_FMA         "ggml: enable FMA"              ${INS_ENB})
if (NOT MSVC)
    option(GGML_F16C    "ggml: enable F16C"             ${INS_ENB}) # in MSVC F16C is implied with AVX2/AVX512
endif()
option(GGML_LASX        "ggml: enable lasx"             ON)
option(GGML_LSX         "ggml: enable lsx"              ON)
option(GGML_SVE         "ggml: enable SVE"              OFF)

if (WIN32)
    set(GGML_WIN_VER "0x602" CACHE STRING "ggml: Windows Version")
endif()

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")

# 3rd party libs / backends
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             OFF)

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_MUSA                            "ggml: use MUSA"                                  OFF)
option(GGML_CUDA_FORCE_DMMV                 "ggml: use dmmv instead of mmvq CUDA kernels"     OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
set   (GGML_CUDA_DMMV_X   "32" CACHE STRING "ggml: x stride for dmmv CUDA kernels")
set   (GGML_CUDA_MMV_Y     "1" CACHE STRING "ggml: y block size for mmv CUDA kernels")
option(GGML_CUDA_F16                        "ggml: use 16 bit floats for some calculations"   OFF)
set   (GGML_CUDA_KQUANTS_ITER "2" CACHE STRING
                                            "ggml: iters./thread per block for Q2_K/Q6_K")
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_USE_GRAPHS                 "ggml: use CUDA graphs (llama.cpp only)"          OFF)

option(GGML_CURL                            "ggml: use libcurl to download model from an URL" OFF)
option(GGML_HIPBLAS                         "ggml: use hipBLAS"                               OFF)
option(GGML_HIP_UMA                         "ggml: use HIP unified memory architecture"       OFF)
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_KOMPUTE                         "ggml: use Kompute"                               OFF)
option(GGML_METAL                           "ggml: use Metal"                                 ${GGML_METAL_DEFAULT})
option(GGML_METAL_NDEBUG                    "ggml: disable Metal debugging"                   OFF)
option(GGML_METAL_SHADER_DEBUG              "ggml: compile Metal with -fno-fast-math"         OFF)
option(GGML_METAL_EMBED_LIBRARY             "ggml: embed Metal library"                       ${GGML_METAL})
set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                            "ggml: metal minimum macOS version")
set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
option(GGML_RPC                             "ggml: use RPC"                                   OFF)
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})
# Debug output
message(STATUS "GGML_SOURCES before filtering: ${GGML_SOURCES}")

# Filter out non-existent files
set(GGML_SOURCES_FILTERED)
foreach(source ${GGML_SOURCES})
    if(EXISTS "${source}")
        list(APPEND GGML_SOURCES_FILTERED ${source})
        message(STATUS "File exists and will be compiled: ${source}")
    else()
        message(WARNING "File does not exist and will be skipped: ${source}")
    endif()
endforeach()

message(STATUS "GGML_SOURCES after filtering: ${GGML_SOURCES_FILTERED}")

# Create a custom target for compiling ggml sources
add_custom_target(ggml_objects)

# Compile each ggml source file
foreach(source ${GGML_SOURCES_FILTERED})
    get_filename_component(source_name ${source} NAME_WE)
    get_filename_component(source_ext ${source} EXT)
    
    if(${source_ext} STREQUAL ".cu")
        set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${source_name}.cu.o)
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_FLAGS}
                    -DGGML_BUILD
                    -DGGML_USE_CUBLAS
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../
                    -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                    -c ${source}
                    -o ${output_file}
            DEPENDS ${source}
            COMMENT "Compiling CUDA file ${source_name}"
        )
    else()
        set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${source_name}.o)
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND ${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS}
                    -DGGML_BUILD
                    -D_GNU_SOURCE
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../
                    -c ${source}
                    -o ${output_file}
            DEPENDS ${source}
            COMMENT "Compiling ${source_name}"
        )
    endif()
    
    list(APPEND GGML_OBJECTS ${output_file})
    add_custom_target(ggml_object_${source_name} DEPENDS ${output_file})
    add_dependencies(ggml_objects ggml_object_${source_name})
endforeach()

# Create the ggml library from the compiled objects
add_library(ggml_library STATIC IMPORTED GLOBAL)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libggml.a
    COMMAND ${CMAKE_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libggml.a ${GGML_OBJECTS}
    COMMAND ${CMAKE_RANLIB} ${CMAKE_CURRENT_BINARY_DIR}/libggml.a
    DEPENDS ggml_objects
    COMMENT "Creating libggml.a"
)
add_custom_target(ggml_library_target DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libggml.a)
add_dependencies(ggml_library ggml_library_target)
set_target_properties(ggml_library PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libggml.a
)

# Link against necessary libraries
target_link_libraries(ggml_library INTERFACE pthread m rt)

if (LLAVA_CUDA)
    target_link_libraries(ggml_library INTERFACE CUDA::cudart CUDA::cublas)
endif()

# Debug output
get_target_property(GGML_SOURCES ggml_library_target SOURCES)
message(STATUS "GGML_SOURCES after target creation: ${GGML_SOURCES}")

# Add the llama library
add_library(llama STATIC
    ../../src/llama.cpp
    ../../include/llama.h
    ../../src/unicode.h
    ../../src/llama-vocab.h
    ../../src/llama-vocab.cpp
    ../../src/unicode.cpp
    ../../src/llama-grammar.cpp
    ../../src/llama-grammar.h
    ../../src/llama-sampling.h
    ../../src/llama-sampling.cpp
    ../../src/unicode-data.cpp
    ../../src/unicode-data.h
    ../../src/llama-impl.h
    ../../common/common.cpp
    ../../common/common.h
)

target_include_directories(llama PUBLIC
    .
    ../..
    ../../common
    ../../include
    ../../ggml/include
    ../../ggml/src
    ../../src
)

target_link_libraries(llama PUBLIC 
    ggml_library
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(llama PRIVATE cxx_std_11)

if (LLAVA_CUDA)
    target_link_libraries(llama PUBLIC CUDA::cudart CUDA::cublas)
endif()

add_dependencies(llama ggml_library_target)

# Add the llava library
add_library(llava STATIC
    ./llava.cpp
    ./llava.h
    ./clip.cpp
    ./clip.h
)

target_include_directories(llava PUBLIC
    .
    ../..
    ../../common
)

target_link_libraries(llava PUBLIC llama ggml_library ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llava PRIVATE cxx_std_11)

if (LLAVA_CUDA)
    target_link_libraries(llava PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Add the server executable
add_executable(llava-server
    llava-server.cpp
)

target_link_libraries(llava-server
    PRIVATE
    llava
    llama
    ggml_library
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    pthread
    m
    rt
)

if (LLAVA_CUDA)
    target_link_libraries(llava-server PRIVATE CUDA::cudart CUDA::cublas)
endif()

target_include_directories(llava-server
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    .
    ../..
    ../../common
    ../../include
    ../../ggml/include
    ../../ggml/src
    ../../src
)

# Installation rules
install(TARGETS llava-server
    RUNTIME DESTINATION bin
)

file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build_info.txt" CONTENT
"C compiler: ${CMAKE_C_COMPILER}
CXX compiler: ${CMAKE_CXX_COMPILER}
CUDA compiler: ${CMAKE_CUDA_COMPILER}
GGML_SOURCES: ${GGML_SOURCES}
")