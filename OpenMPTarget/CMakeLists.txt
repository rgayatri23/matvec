cmake_minimum_required (VERSION 3.12)
project(matvec
  LANGUAGES C CXX
  )

# don't allow in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(STATUS "Warning! Building from the source directory is not allow")
  message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build from a separate directory")
  message(ERROR "In-source build")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CXX_STANDARD 17)

SET(MyTarget matvec)

ADD_EXECUTABLE(
  ${MyTarget}
  ${CMAKE_SOURCE_DIR}/matvec.cpp
  ${CMAKE_SOURCE_DIR}/arrayMD.h
  )

if(${OPENMP})
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(${OPENMP_TARGET})
  add_compile_definitions(OPENMP_TARGET=-DOPENMP_TARGET)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    add_compile_options(-fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_PATH} -I${CUDA_PATH}/include -ffp-contract=fast)
    target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()
