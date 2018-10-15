# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

# keep two libraries: libsirius and libsirius_f

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius_cu "${CUFILES_KERNELS};${CUFILES_SDDK}")
  set_target_properties(sirius_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  install(TARGETS sirius_cu ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()
if(CREATE_FORTRAN_BINDINGS)
  add_library(sirius_f "sirius_api.cpp;sirius.f90")
  install(TARGETS sirius_f ARCHIVE DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/)
  set_target_properties(sirius_f PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(sirius_f PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  install( CODE
    "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/mod_files\" \"${CMAKE_INSTALL_PREFIX}/fortran\")"
    )
endif()

if(CREATE_FORTRAN_BINDINGS)
  add_library(sirius_shared SHARED "sirius_api.cpp;sirius.f90;${CUFILES_KERNELS};${CUFILES_SDDK}")
  set_target_properties(sirius_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(sirius_shared PROPERTIES OUTPUT_NAME sirius)
  set_target_properties(sirius_shared PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  install(TARGETS sirius_shared LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
elseif(USE_CUDA)
  add_library(sirius_shared SHARED "${CUFILES_KERNELS};${CUFILES_SDDK}")
  set_target_properties(sirius_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(sirius_shared PROPERTIES OUTPUT_NAME sirius)
  install(TARGETS sirius_shared LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()


install(DIRECTORY ./ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius" FILES_MATCHING REGEX ".*(hpp|h)$")
