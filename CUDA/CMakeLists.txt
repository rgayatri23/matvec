cmake_minimum_required (VERSION 3.12)
project(matvec
  LANGUAGES C CXX CUDA
  )
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

# don't allow in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(STATUS "Warning! Building from the source directory is not allow")
  message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build from a separate directory")
  message(ERROR "In-source build")
endif()

SET(MyTarget matvec)

find_package(CUDA REQUIRED)
message(STATUS "CUDA ROOT DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

#SET(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#SET(CMAKE_VERBOSE_MAKEFILE ON)
#set_source_files_properties(${CMAKE_SOURCE_DIR}/matvec.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${CUDA_TOOLKIT_ROOT_DIR}/lib64")

#PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
ADD_EXECUTABLE(
  ${MyTarget}
  ${CMAKE_SOURCE_DIR}/matvec.cpp
  ${CMAKE_SOURCE_DIR}/matvec.h
  ${CMAKE_SOURCE_DIR}/arrayMD.h
  ${CMAKE_SOURCE_DIR}/arrayMDgpu.h
  )

SET_SOURCE_FILES_PROPERTIES(${CMAKE_SOURCE_DIR}/matvec.cpp PROPERTIES LANGUAGE CUDA)
SET_TARGET_PROPERTIES(${MyTarget} PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(${MyTarget} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_include_directories(${MyTarget} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
set(CUDA_CXX_FLAGS "${CUDA_CXX_FLAGS} --cuda-gpu-arch=sm_70")
