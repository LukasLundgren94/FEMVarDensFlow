cmake_minimum_required(VERSION 3.5)

project(ocean)

# Include directories required to build
include_directories(SYSTEM ${DOLFIN_INCLUDE_DIRS}
  ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Find and include DOLFIN
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

#
set(PROJECT_DIR ./)
set(INSTALL_INCLUDE_DIR ${PROJECT_DIR}/local/include/${PROJECT_NAME})
# set(INSTALL_UFL_INCLUDE_DIR ${PROJECT_DIR}/local/include/ufl)
set(INSTALL_LIB_DIR ${PROJECT_DIR}/local/lib)

# set the source files:
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UFL_DIR ${SRC_DIR}/ufl/*)
set(${PROJECT_NAME}_INCLUDE_DIR src)

file(GLOB ${PROJECT_NAME}_SOURCES ${SRC_DIR}/*/*.cpp)
file(GLOB ${PROJECT_NAME}_HEADERS ${SRC_DIR}/*/*.h)
file(GLOB ${PROJECT_NAME}_UFL_SOURCES ${UFL_DIR}/*/*.cpp)
file(GLOB ${PROJECT_NAME}_UFL_HEADERS ${UFL_DIR}/*/*.h)

add_library(${PROJECT_NAME} STATIC
  ${${PROJECT_NAME}_SOURCES}
  ${${PROJECT_NAME}_HEADERS}
  ${${PROJECT_NAME}_UFL_SOURCES}
  ${${PROJECT_NAME}_UFL_HEADERS}
  )

#
target_link_libraries(${PROJECT_NAME}
  LINK_PUBLIC core
  ${DOLFIN_LIBRARIES}
  ${DOLFIN_3RD_PARTY_LIBRARIES})

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION ${INSTALL_LIB_DIR})

# install header files
install(DIRECTORY src/ DESTINATION  ${INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h")
