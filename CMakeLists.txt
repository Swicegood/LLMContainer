cmake_minimum_required(VERSION 3.14)
project(llava-server CUDA CXX)

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
    enable_language(CUDA)   # Method 1: Using PkgConfig
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(CUBLAS cublas IMPORTED_TARGET)
        if(CUBLAS_FOUND)
            set(CUDA_LIBRARIES PkgConfig::cublas)
        endif()
    endif()
    
    # Method 2: Using FindCUDAToolkit
    if(NOT CUBLAS_FOUND)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_LIBRARIES CUDA::cudart CUDA::cublas)
        
        # Additional settings for SDK environments
        set(CUDAToolkit_TARGET_DIR "${CUDA_TOOLKIT_TARGET_DIR}" CACHE PATH "" FORCE)
        find_library(CUDA_CUDART NAMES cudart HINTS ${CUDA_TOOLKIT_TARGET_DIR} PATH_SUFFIXES lib64 lib)
    endif()
    
    add_definitions(-DGGML_USE_CUBLAS)
endif()

# Add nlohmann/json
include_directories(./include)

# Force include all ggml source files
set(GGML_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-alloc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-backend.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-quants.c
)

if(LLAVA_CUDA)
    list(APPEND GGML_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cuda.cu)
endif()

# Debug output
message(STATUS "GGML_SOURCES: ${GGML_SOURCES}")
foreach(source ${GGML_SOURCES})
    if(EXISTS "${source}")
        message(STATUS "File exists: ${source}")
    else()
        message(WARNING "File does not exist: ${source}")
    endif()
endforeach()

add_library(ggml STATIC ${GGML_SOURCES})

set_target_properties(ggml PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(ggml PUBLIC GGML_SHARED)
target_compile_options(ggml PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-fPIC>
    $<$<COMPILE_LANGUAGE:C>:-fvisibility=default>
    $<$<COMPILE_LANGUAGE:C>:-Wno-unused-function>
    $<$<COMPILE_LANGUAGE:C>:-Wno-unused-variable>
)

if (LLAVA_CUDA)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cuda.cu PROPERTIES LANGUAGE CUDA)
    target_compile_options(ggml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

target_include_directories(ggml PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
)

if (LLAVA_CUDA)
    target_link_libraries(ggml PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Debug output
get_target_property(GGML_SOURCES ggml SOURCES)
message(STATUS "GGML_SOURCES after target creation: ${GGML_SOURCES}")

# Rest of the CMakeLists.txt remains the same...
# Add the llama library
add_library(llama STATIC
    ../../src/llama.cpp
    ../../include/llama.h
)

target_include_directories(llama PUBLIC
    .
    ../..
    ../../common
    ../../include
    ../../ggml/include
    ../../ggml/src
)

target_link_libraries(llama PUBLIC ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama PRIVATE cxx_std_11)

if (LLAVA_CUDA)
    target_link_libraries(llama PUBLIC CUDA::cudart CUDA::cublas)
endif()

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

target_link_libraries(llava PUBLIC llama ggml ${CMAKE_THREAD_LIBS_INIT})
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
    ggml
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
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
)

# Installation rules
install(TARGETS llava-server
    RUNTIME DESTINATION bin
)