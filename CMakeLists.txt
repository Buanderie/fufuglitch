cmake_minimum_required(VERSION 2.8)
project( FUFUGLITCH )

#
#
## CHECK FOR COMPILER CAPABILITIES
#
#
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
##

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread -fopenmp" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3" )

find_package( OpenCV REQUIRED )

# sources
file (GLOB_RECURSE ALL_SRC ${PROJECT_SOURCE_DIR}/*.cc)
file (GLOB_RECURSE ALL_HEADERS ${PROJECT_SOURCE_DIR}/*.h)

add_executable( fufu ${ALL_SRC} ${ALL_HEADERS} )
target_link_libraries( fufu ${OpenCV_LIBS} opencv_world )
