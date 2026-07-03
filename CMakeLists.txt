cmake_minimum_required(VERSION 3.17...3.30)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

set(PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/graph_layout")

# Cython: generate _speedups.c from _speedups.pyx
# Directives boundscheck/wraparound/cdivision/language_level are set in the
# .pyx header; embedsignature is enabled here for parity with the old build.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_speedups.c
    COMMAND Python::Interpreter -m cython
        -3
        --directive embedsignature=True
        "${PKG_DIR}/_speedups.pyx"
        -o ${CMAKE_CURRENT_BINARY_DIR}/_speedups.c
    DEPENDS
        ${PKG_DIR}/_speedups.pyx
        ${PKG_DIR}/_speedups.pxd
    COMMENT "Generating _speedups.c from _speedups.pyx"
)

# Build Python extension module
python_add_library(_speedups MODULE
    ${CMAKE_CURRENT_BINARY_DIR}/_speedups.c
    WITH_SOABI
)

# Optimization flags (mirrors the previous setuptools configuration)
if(MSVC)
    target_compile_options(_speedups PRIVATE /O2)
else()
    target_compile_options(_speedups PRIVATE -O3 -std=c99)
endif()

# Install alongside the package sources
install(TARGETS _speedups DESTINATION graph_layout)
