cmake_minimum_required (VERSION 3.12)
project(matvec
  LANGUAGES C CXX
  )
set(CMAKE_BUILD_TYPE Release)
set(CXX_STANDARD 17)

# don't allow in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(STATUS "Warning! Building from the source directory is not allow")
  message(STATUS "Remove 'CMakeCache.txt' and 'CMakeFiles' and build from a separate directory")
  message(ERROR "In-source build")
endif()

SET(MyTarget matvec)

ADD_EXECUTABLE(
  ${MyTarget}
  ${CMAKE_SOURCE_DIR}/matvec.cpp
  ${CMAKE_SOURCE_DIR}/arrayMD.h
  )
