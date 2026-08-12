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
option(LEVO_BUILD_PARITY_TOOLS "Build local real-model parity tools" OFF)

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
    src/levo-audio-ops.cpp
    src/levo-conditioner.cpp
    src/levo-flow-conditioning.cpp
    src/levo-flow-model.cpp
    src/levo-generator.cpp
    src/levo-kv.cpp
    src/levo-lm.cpp
    src/levo-model.cpp
    src/levo-pattern.cpp
    src/levo-renderer-pattern.cpp
    src/levo-sampling.cpp
    src/levo-token-io.cpp
    src/levo-tokenizer.cpp
    src/levo-vae-model.cpp
    src/levo-vae.cpp
    src/levo-wav.cpp
    src/levo.cpp
    third_party/llama-unicode/unicode.cpp
    third_party/llama-unicode/unicode-data.cpp
    third_party/utf8proc/utf8proc.c
)
add_library(levo::core ALIAS levo-core)

target_include_directories(levo-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-unicode
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc
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

if(LEVO_BUILD_PARITY_TOOLS)
    add_executable(levo-parity tools/levo-parity.cpp)
    target_link_libraries(levo-parity PRIVATE levo::core)
    target_include_directories(levo-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_executable(levo-condition-parity tools/levo-condition-parity.cpp)
    target_link_libraries(levo-condition-parity PRIVATE levo::core)
    target_include_directories(levo-condition-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_executable(levo-cfg-parity tools/levo-cfg-parity.cpp)
    target_link_libraries(levo-cfg-parity PRIVATE levo::core)
    target_include_directories(levo-cfg-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_executable(levo-tokenizer-parity tools/levo-tokenizer-parity.cpp)
    target_link_libraries(levo-tokenizer-parity PRIVATE levo::core)
    target_include_directories(levo-tokenizer-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_executable(levo-vae-parity tools/levo-vae-parity.cpp)
    target_link_libraries(levo-vae-parity PRIVATE levo::core)
    target_include_directories(levo-vae-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

set_target_properties(levo-core levo-cli PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(LEVO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
