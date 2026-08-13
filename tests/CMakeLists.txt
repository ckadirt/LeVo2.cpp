add_executable(test-backend-smoke test-backend-smoke.cpp)
target_link_libraries(test-backend-smoke PRIVATE levo::core)

add_executable(test-cantor-abi test-cantor-abi.c)
target_link_libraries(test-cantor-abi PRIVATE levo::cantor-engine)
add_test(NAME cantor-abi COMMAND test-cantor-abi)

add_test(NAME backend-smoke-cpu COMMAND test-backend-smoke cpu)
if(GGML_CUDA)
    add_test(NAME backend-smoke-cuda COMMAND test-backend-smoke cuda)
    set_tests_properties(backend-smoke-cuda PROPERTIES LABELS "cuda")
endif()

foreach(test_name IN ITEMS checkpoint model lm kv conditioner flow-model flow-conditioning cuda-precision flow-estimator flow-renderer generation pattern renderer-pattern render sampling tokenizer token-io audio-ops vae-model vae wav progress)
    add_executable(test-${test_name} test-${test_name}.cpp)
    target_link_libraries(test-${test_name} PRIVATE levo::core)
    target_include_directories(test-${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME ${test_name} COMMAND test-${test_name})
endforeach()

target_include_directories(test-progress PRIVATE ${PROJECT_SOURCE_DIR}/tools)

if(GGML_CUDA)
    add_test(NAME audio-ops-cuda COMMAND test-audio-ops cuda)
    set_tests_properties(audio-ops-cuda PROPERTIES LABELS "cuda")
    add_test(NAME cuda-precision-cuda COMMAND test-cuda-precision cuda)
    set_tests_properties(cuda-precision-cuda PROPERTIES LABELS "cuda;renderer")
endif()

if(GGML_CUDA)
    add_test(NAME vae-cuda COMMAND test-vae cuda)
    set_tests_properties(vae-cuda PROPERTIES LABELS "cuda;renderer")
endif()

target_compile_definitions(test-tokenizer PRIVATE
    LEVO_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_compile_definitions(test-conditioner PRIVATE
    LEVO_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
