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
#
# -ffp-contract=off / /fp:precise: the pure-Python fallbacks in
# graph_layout.force must reproduce the compiled kernels bit for bit, and the
# force-directed iterations amplify a single-ulp difference into a visibly
# different drawing over a few hundred steps. Clang on arm64 macOS contracts
# `dx * dx + dy * dy` into an FMA by default, which CPython's float arithmetic
# never does, so contraction has to be disabled for the two paths to agree.
if(MSVC)
    target_compile_options(_speedups PRIVATE /O2 /fp:precise)
else()
    target_compile_options(_speedups PRIVATE -O3 -std=c99 -ffp-contract=off)
endif()

# Install alongside the package sources
install(TARGETS _speedups DESTINATION graph_layout)
