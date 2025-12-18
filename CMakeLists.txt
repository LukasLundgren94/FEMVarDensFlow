cmake_minimum_required(VERSION 3.5)

project(fem_varden_flow)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(FEM_VARDEN_FLOW_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/local/include)
set(FEM_VARDEN_FLOW_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/local/bin)
set(FEM_VARDEN_FLOW_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/local/lib)
set(FEM_VARDEN_FLOW_HEADERS_DIR ${PROJECT_SOURCE_DIR}/core/src)

include_directories(${FEM_VARDEN_FLOW_INSTALL_INCLUDE_DIR})
include_directories(${FEM_VARDEN_FLOW_HEADERS_DIR})

add_subdirectory(core)
#add_subdirectory(nse)

#add_subdirectory(nse_ac)
add_subdirectory(ocean)
add_subdirectory(varden)
