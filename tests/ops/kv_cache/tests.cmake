ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/append_main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/append_full.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/append_prefix.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)

# Fork-local (rtx4090-port): the shipped E8 root codec header, header-only.
ninfer_add_op_test(ninfer_e8_root_codec_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/e8_root_codec.cu"
  LIBRARIES ninfer_ops)

# Fork-local (rtx4090-port): isolated E8 lattice KV codec testbed. It carries its own
# reference implementation of the E8 mathematics and does not include
# src/ops/kv_cache/kernels/e8_root_codec.cuh, so it is an algorithm oracle, not coverage of the
# shipped codec. ninfer_e8_root_codec_test drives the production header.
add_executable(ninfer_test_e8_codec
  "${PROJECT_SOURCE_DIR}/tools/test_kv/verify_1m_retrieval.cu"
  "${PROJECT_SOURCE_DIR}/tools/test_kv/test_e8_codec.cu")
target_include_directories(ninfer_test_e8_codec PRIVATE "${PROJECT_SOURCE_DIR}/tools/test_kv")
target_link_libraries(ninfer_test_e8_codec PRIVATE CUDA::cudart)
add_test(NAME ninfer_test_e8_codec COMMAND ninfer_test_e8_codec)
set_tests_properties(ninfer_test_e8_codec PROPERTIES SKIP_RETURN_CODE 77)
