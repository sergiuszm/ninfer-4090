target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/act_quant_g64.cu"
  "${CMAKE_CURRENT_LIST_DIR}/int8_proj_launch.cu"
)
