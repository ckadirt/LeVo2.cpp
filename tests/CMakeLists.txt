add_executable(test-backend-smoke test-backend-smoke.cpp)
target_link_libraries(test-backend-smoke PRIVATE levo::core)

add_test(NAME backend-smoke-cpu COMMAND test-backend-smoke cpu)
if(GGML_CUDA)
    add_test(NAME backend-smoke-cuda COMMAND test-backend-smoke cuda)
    set_tests_properties(backend-smoke-cuda PROPERTIES LABELS "cuda")
endif()
