cmake_minimum_required(VERSION 3.18)

project(LeVo2
    VERSION 0.1.0
    DESCRIPTION "Portable LeVo 2 LeLM inference with GGML"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LEVO_BUILD_TESTS "Build LeVo2.cpp tests" ON)

set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_SERVER OFF CACHE BOOL "" FORCE)
if(GGML_CUDA AND NOT DEFINED GGML_CUDA_NCCL)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
endif()
if(GGML_CUDA AND NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)
find_package(Threads REQUIRED)

add_library(levo-core STATIC
    src/levo.cpp
)
add_library(levo::core ALIAS levo-core)

target_include_directories(levo-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(levo-core PUBLIC ggml::ggml Threads::Threads)
target_compile_features(levo-core PUBLIC cxx_std_17)
target_compile_definitions(levo-core PUBLIC LEVO_VERSION="${PROJECT_VERSION}")
if(GGML_CUDA)
    target_compile_definitions(levo-core PUBLIC LEVO_HAS_CUDA=1)
else()
    target_compile_definitions(levo-core PUBLIC LEVO_HAS_CUDA=0)
endif()

if(MSVC)
    target_compile_options(levo-core PRIVATE /W4)
else()
    target_compile_options(levo-core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(levo-cli tools/levo-cli.cpp)
target_link_libraries(levo-cli PRIVATE levo::core)

set_target_properties(levo-core levo-cli PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(LEVO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
