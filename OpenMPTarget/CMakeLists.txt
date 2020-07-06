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
  ${CMAKE_SOURCE_DIR}/../arrayMD/arrayMD.h
  )

if(${OPENMP})
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(${OPENMP_TARGET})
  find_package(CUDA REQUIRED)
  find_package(OpenMP REQUIRED)
  target_compile_definitions(${MyTarget} PRIVATE OPENMP_TARGET)
  target_compile_options(${MyTarget} PRIVATE -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include -ffp-contract=fast)
  target_link_libraries(${MyTarget} PUBLIC OpenMP::OpenMP_CXX)
  target_link_options(${MyTarget} PRIVATE -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp=libiomp5)
endif()
