# Require CMake 3.5
cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME demo)
project(${PROJECT_NAME})

set(FEM_VARDEN_FLOW_ROOT ../../..)
set(PROJECT_LIB_DIR ${FEM_VARDEN_FLOW_ROOT}/local/lib)
set(FEM_VARDEN_FLOW_INCLUDE_DIRS ${FEM_VARDEN_FLOW_ROOT}/local/include)

# find dependiency libs
find_library(CORE_LIB core ${PROJECT_LIB_DIR})
find_library(VARDEN_LIB varden ${PROJECT_LIB_DIR})
find_library(OCEAN_LIB ocean ${PROJECT_LIB_DIR})
include_directories(${FEM_VARDEN_FLOW_INCLUDE_DIRS})

find_package(DOLFIN)
include(${DOLFIN_USE_FILE})

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# C++11 compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 HAVE_STD_CPP11)
if (HAVE_STD_CPP11)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  CHECK_CXX_COMPILER_FLAG(-std=c++0x HAVE_STD_CPP0x)
  if (HAVE_STD_CPP0x)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  endif()
endif()

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Include directories
include_directories(${FEM_CLAW_INCLUDE_DIRS})
include_directories(SYSTEM ${DOLFIN_INCLUDE_DIRS}
  ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME}
  main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME}
  ${OCEAN_LIB}
  ${VARDEN_LIB}
  ${CORE_LIB}
  ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})
