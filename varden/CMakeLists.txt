cmake_minimum_required(VERSION 3.5)
project(varden)

# Include directories required to build
include_directories(SYSTEM ${DOLFIN_INCLUDE_DIRS}
  ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Find and include DOLFIN
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

#
set(PROJECT_DIR ./)
set(INSTALL_INCLUDE_DIR ${PROJECT_DIR}/local/include/${PROJECT_NAME})
set(INSTALL_LIB_DIR ${PROJECT_DIR}/local/lib)

# set the source files:

# PCS files
set(SRC_DIR_PCS ${PROJECT_SOURCE_DIR}/src/pcs)
set(UFL_DIR_PCS ${SRC_DIR_PCS}/ufl)

file(GLOB ${PROJECT_NAME}_SOURCES_PCS ${SRC_DIR_PCS}/*.cpp)
file(GLOB ${PROJECT_NAME}_HEADERS_PCS ${SRC_DIR_PCS}/*.h)
file(GLOB ${PROJECT_NAME}_UFL_SOURCES_PCS ${UFL_DIR_PCS}/*.cpp)
file(GLOB ${PROJECT_NAME}_UFL_HEADERS_PCS ${UFL_DIR_PCS}/*.h)

# AC files
set(SRC_DIR_AC ${PROJECT_SOURCE_DIR}/src/ac)
set(UFL_DIR_AC ${SRC_DIR_AC}/ufl)

file(GLOB ${PROJECT_NAME}_SOURCES_AC ${SRC_DIR_AC}/*.cpp)
file(GLOB ${PROJECT_NAME}_HEADERS_AC ${SRC_DIR_AC}/*.h)
file(GLOB ${PROJECT_NAME}_UFL_SOURCES_AC ${UFL_DIR_AC}/*.cpp)
file(GLOB ${PROJECT_NAME}_UFL_HEADERS_AC ${UFL_DIR_AC}/*.h)

# AC_TC files
set(SRC_DIR_AC_TC ${PROJECT_SOURCE_DIR}/src/solver)
set(UFL_DIR_AC_TC ${SRC_DIR_AC_TC}/ufl)

file(GLOB ${PROJECT_NAME}_SOURCES_AC_TC
  ${SRC_DIR_AC_TC}/*.cpp
  ${SRC_DIR_AC_TC}/troubled_cells_boundary/*.cpp)
file(GLOB ${PROJECT_NAME}_HEADERS_AC_TC
  ${SRC_DIR_AC_TC}/*.h
  ${SRC_DIR_AC_TC}/troubled_cells_boundary/*.h)
file(GLOB ${PROJECT_NAME}_UFL_SOURCES_AC_TC
  ${UFL_DIR_AC_TC}/*.cpp
  ${UFL_DIR_AC_TC}/troubled_cells_boundary/*.cpp)
file(GLOB ${PROJECT_NAME}_UFL_HEADERS_AC_TC
  ${UFL_DIR_AC_TC}/*.h
  ${UFL_DIR_AC_TC}/troubled_cells_boundary/*.h)

# Common files
set(SRC_DIR_COMMON ${PROJECT_SOURCE_DIR}/src/common)
set(UFL_DIR_COMMON ${SRC_DIR_COMMON}/ufl)

file(GLOB ${PROJECT_NAME}_SOURCES_COMMON ${SRC_DIR_COMMON}/*.cpp)
file(GLOB ${PROJECT_NAME}_HEADERS_COMMON ${SRC_DIR_COMMON}/*.h)
file(GLOB ${PROJECT_NAME}_UFL_SOURCES_COMMON ${UFL_DIR_COMMON}/*.cpp)
file(GLOB ${PROJECT_NAME}_UFL_HEADERS_COMMON ${UFL_DIR_COMMON}/*.h)

# Now COMMON is used since it is added first...
add_library(${PROJECT_NAME} STATIC
  ${${PROJECT_NAME}_SOURCES_COMMON}
  ${${PROJECT_NAME}_HEADERS_COMMON}
  ${${PROJECT_NAME}_UFL_SOURCES_COMMON}
  ${${PROJECT_NAME}_UFL_HEADERS_COMMON}
  ${${PROJECT_NAME}_SOURCES_AC}
  ${${PROJECT_NAME}_HEADERS_AC}
  ${${PROJECT_NAME}_UFL_SOURCES_AC}
  ${${PROJECT_NAME}_UFL_HEADERS_AC}
  ${${PROJECT_NAME}_SOURCES_AC_TC}
  ${${PROJECT_NAME}_HEADERS_AC_TC}
  ${${PROJECT_NAME}_UFL_SOURCES_AC_TC}
  ${${PROJECT_NAME}_UFL_HEADERS_AC_TC}
  ${${PROJECT_NAME}_SOURCES_PCS}
  ${${PROJECT_NAME}_HEADERS_PCS}
  ${${PROJECT_NAME}_UFL_SOURCES_PCS}
  ${${PROJECT_NAME}_UFL_HEADERS_PCS}
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ocean/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ocean/src/meltparameterization>
    $<INSTALL_INTERFACE:local/include>
    $<INSTALL_INTERFACE:local/include/ocean/meltparameterization>
  )

target_link_libraries(${PROJECT_NAME}
  LINK_PUBLIC core ocean
  ${DOLFIN_LIBRARIES}
  ${DOLFIN_3RD_PARTY_LIBRARIES})

# install the lib 
install(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_LIB_DIR})

# install header files
install(DIRECTORY src/ DESTINATION  ${INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h")
