# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.30)

option(NPPDX_BUILD_CUDA_EXAMPLES "Build CUDA examples" ON)
option(NPPDX_BUILD_CLANG_PTX_EXAMPLES "Build Clang device-only PTX examples with CUDA Driver API runners" OFF)
option(NPPDX_INSTALL_EXAMPLES "Install compiled example binaries" OFF)

set(NPPDX_EXAMPLE_PROJECT_LANGUAGES CXX)
if(NPPDX_BUILD_CUDA_EXAMPLES)
    list(APPEND NPPDX_EXAMPLE_PROJECT_LANGUAGES CUDA)
endif()

# NPPDx project
project(nppdx_examples VERSION 0.1.1 LANGUAGES ${NPPDX_EXAMPLE_PROJECT_LANGUAGES})

if(PROJECT_IS_TOP_LEVEL)
    # Enable running examples with CTest
    enable_testing()

    set(DX_PROJECT_CMAKE_VAR_PREFIX "NPPDX")
    set(DX_PROJECT_FULL_NAME "NPPDx")

    # ******************************************
    # NPPDx Install Dirs
    # ******************************************
    include(GNUInstallDirs)
    set(NPPDX_EXAMPLES_BIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/example/")

    # Include custom CMake modules/scripts
    list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/
    )

    if(NOT MSVC)
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -Wno-deprecated-declarations")
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -m64")
        endif ()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1.0)
                set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} --diag_suppress1")
                set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} --diag_suppress111")
                set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} --diag_suppress177")
                set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} --diag_suppress941")
            # endif()
        endif()

        # because -Wall is passed, all diagnostic ignores of -Wunkown-pragmas
        # are ignored, which leads to unlegible CuTe compilation output
        # fixed in GCC13 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
        # but still not widely adopted
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")


        # Clang
        if(NPPDX_BUILD_CUDA_EXAMPLES AND CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*")
            # clang complains about unused function in CUDA system headers
            set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} -Wno-unused-function")
        endif()
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(NPPDX_CUDA_CXX_FLAGS "${NPPDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NPPDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    if(NPPDX_BUILD_CUDA_EXAMPLES)
        set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${NPPDX_CUDA_CXX_FLAGS}\"")
    endif()

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(NPPDX_CUDA_ARCHITECTURES 70-real;80-real CACHE
        STRING "List of targeted NPPDx CUDA architectures, for example \"70-real;75-real;80\""
    )
    # Remove unsupported architectures
    list(REMOVE_ITEM NPPDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53)
    list(REMOVE_ITEM NPPDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;)
    list(REMOVE_ITEM NPPDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual)
    message(STATUS "Targeted NPPDx CUDA Architectures: ${NPPDX_CUDA_ARCHITECTURES}")

    if(NPPDX_BUILD_CUDA_EXAMPLES OR NPPDX_BUILD_CLANG_PTX_EXAMPLES)
        set(NPPDX_FIND_PACKAGE_MODE REQUIRED)
    else()
        set(NPPDX_FIND_PACKAGE_MODE QUIET)
    endif()
    find_package(nppdx ${NPPDX_FIND_PACKAGE_MODE} CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.."
    )
    if(nppdx_FOUND)
        message(STATUS "Found nppdx package")
    endif()
    if(NPPDX_BUILD_CLANG_PTX_EXAMPLES
        AND NOT TARGET nppdx::nppdx
        AND NOT nppdx_INCLUDE_DIRS
        AND NOT nppdx_INCLUDE_DIR)
        get_filename_component(NPPDX_INSTALLED_PREFIX "${PROJECT_SOURCE_DIR}/../.." ABSOLUTE)
        if(EXISTS "${NPPDX_INSTALLED_PREFIX}/include/nppdx.hpp")
            set(nppdx_INCLUDE_DIR "${NPPDX_INSTALLED_PREFIX}/include")
            set(nppdx_INCLUDE_DIRS "${NPPDX_INSTALLED_PREFIX}/include")
        else()
            message(FATAL_ERROR
                "NPPDX_BUILD_CLANG_PTX_EXAMPLES requires either an nppdx package target or "
                "nppdx_INCLUDE_DIRS pointing at installed NPPDx headers")
        endif()
    endif()
endif()

if(NPPDX_BUILD_CUDA_EXAMPLES
    AND NOT TARGET nppdx::nppdx)
    message(FATAL_ERROR
        "NPPDX_BUILD_CUDA_EXAMPLES requires an nppdx::nppdx target. "
        "Set CMAKE_PREFIX_PATH to the package root, set nppdx_DIR to the package config directory, "
        "or disable NPPDX_BUILD_CUDA_EXAMPLES.")
endif()

if(NPPDX_BUILD_CLANG_PTX_EXAMPLES
    AND NOT TARGET nppdx::nppdx
    AND NOT nppdx_INCLUDE_DIRS
    AND NOT nppdx_INCLUDE_DIR)
    message(FATAL_ERROR
        "NPPDX_BUILD_CLANG_PTX_EXAMPLES requires either an nppdx package target or "
        "nppdx_INCLUDE_DIRS pointing at installed NPPDx headers")
endif()

if(NPPDX_BUILD_CUDA_EXAMPLES)
    find_package(CUDAToolkit REQUIRED)
endif()

if(NPPDX_BUILD_CLANG_PTX_EXAMPLES)
    if(NOT NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR)
        message(FATAL_ERROR
            "NPPDX_BUILD_CLANG_PTX_EXAMPLES requires NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR to point to a directory "
            "containing cuda_runtime.h")
    endif()
    if(NOT EXISTS "${NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR}/cuda_runtime.h")
        message(FATAL_ERROR
            "NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR='${NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR}' does not contain "
            "cuda_runtime.h")
    endif()
    if(NOT COMMAND nppdx_cmake_arch_to_clang_arch)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            message(FATAL_ERROR
                "NPPDX_BUILD_CLANG_PTX_EXAMPLES requires CMAKE_CXX_COMPILER to be Clang. "
                "Current CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}', "
                "CMAKE_CXX_COMPILER_ID='${CMAKE_CXX_COMPILER_ID}'")
        endif()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 21)
            message(FATAL_ERROR
                "NPPDX_BUILD_CLANG_PTX_EXAMPLES requires Clang 21 or newer. "
                "Current CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}', "
                "CMAKE_CXX_COMPILER_VERSION='${CMAKE_CXX_COMPILER_VERSION}'")
        endif()

        set(NPPDX_CLANG_PTX_COMPILER "${CMAKE_CXX_COMPILER}")
        message(STATUS "NPPDx Clang PTX compiler: ${NPPDX_CLANG_PTX_COMPILER}")

        function(nppdx_cmake_arch_to_clang_arch OUT_VAR CUDA_ARCH)
            string(REGEX MATCH "^[0-9]+" NPPDX_CLANG_ARCH_NUMBER "${CUDA_ARCH}")
            if(NOT NPPDX_CLANG_ARCH_NUMBER)
                message(FATAL_ERROR "Unable to parse CUDA architecture '${CUDA_ARCH}'")
            endif()
            set(${OUT_VAR} "sm_${NPPDX_CLANG_ARCH_NUMBER}" PARENT_SCOPE)
        endfunction()
    endif()
endif()

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${NPPDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")
    add_compile_definitions(NPPDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

set(NPPDX_EXAMPLE_DATA_ENV "NPPDX_EXAMPLE_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/common/images")

# ###############################################################
# install_example
# ###############################################################
function(install_example EXAMPLE_TARGET)
    install(TARGETS ${EXAMPLE_TARGET}
        RUNTIME DESTINATION ${NPPDX_EXAMPLES_BIN_INSTALL_DIR}/
        COMPONENT Examples
    )
endfunction()

###############################################################
# Common example objects
# #############################################################
if(NPPDX_BUILD_CUDA_EXAMPLES)
    set(common_example_objects_sources common/error_checking.cpp)
    set_source_files_properties("${common_example_objects_sources}" PROPERTIES LANGUAGE CUDA)
    add_library(common_example_objects STATIC "${common_example_objects_sources}")
    target_link_libraries(common_example_objects PRIVATE
        nppdx::nppdx)

    target_compile_definitions(common_example_objects PRIVATE NPPDX_TEST_NO_GTEST)
    set_target_properties(common_example_objects
        PROPERTIES
            CUDA_ARCHITECTURES "${NPPDX_CUDA_ARCHITECTURES}"
    )
endif()


# ###############################################################
# add_nppdx_example
# ###############################################################
function(add_nppdx_example_impl GROUP_TARGET EXAMPLE_SOURCES DO_INSTALL)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    #Compute the test name also. It starts with nppdx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "nppdx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            nppdx::nppdx
            common_example_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND $<PATH:RELATIVE_PATH,$<TARGET_FILE:${EXAMPLE_TARGET}>,${CMAKE_CURRENT_BINARY_DIR}>)
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${NPPDX_CUDA_ARCHITECTURES}"
    )

    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            # Keep compression target-local for standalone and source-tree builds.
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )

    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
            ENVIRONMENT "${NPPDX_EXAMPLE_DATA_ENV}"
    )

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})

    if(DO_INSTALL)
        install_example(${EXAMPLE_TARGET})
    endif()
endfunction()

function(add_nppdx_example GROUP_TARGET  EXAMPLE_SOURCES)
    add_nppdx_example_impl("${GROUP_TARGET}" "${EXAMPLE_SOURCES}" "${NPPDX_INSTALL_EXAMPLES}")
endfunction()


# Wrapper function for examples with NPP comparison
function(add_nppdx_example_with_npp_impl GROUP_TARGET EXAMPLE_SOURCES DO_INSTALL)
    add_nppdx_example_impl("${GROUP_TARGET}" "${EXAMPLE_SOURCES}" "${DO_INSTALL}")

    # Get target name from source
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    # Find and link NPP
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::nppc CUDA::nppif CUDA::nppig CUDA::nppicc)
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE NPPDX_EXAMPLE_HAS_NPP)
endfunction()

function(add_nppdx_example_with_npp GROUP_TARGET EXAMPLE_SOURCES)
    add_nppdx_example_with_npp_impl("${GROUP_TARGET}" "${EXAMPLE_SOURCES}" "${NPPDX_INSTALL_EXAMPLES}")
endfunction()

# ###############################################################
# add_nppdx_nvrtc_example
# ###############################################################
function(add_nppdx_nvrtc_example GROUP_TARGET EXAMPLE_SOURCES)
    find_package(CUDAToolkit REQUIRED)

    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
     # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")

    #Compute the test name also. It starts with nppdx.example.+subdirectory.+test source name
    set(EXAMPLE_NAME "nppdx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            nppdx::nppdx
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
    )
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
            COMMONDX_INCLUDE_DIR="${nppdx_commondx_INCLUDE_DIR}"
            NPPDX_INCLUDE_DIRS="${nppdx_INCLUDE_DIRS}"
    )
    # Disable NVRTC examples for NVC++ compiler, because of libcu++/NRTC/NVC++ bug
    if(NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*")))
        add_test(NAME ${EXAMPLE_NAME} COMMAND $<PATH:RELATIVE_PATH,$<TARGET_FILE:${EXAMPLE_TARGET}>,${CMAKE_CURRENT_BINARY_DIR}>)
        set_tests_properties(${EXAMPLE_NAME}
            PROPERTIES
                LABELS "EXAMPLE"
                ENVIRONMENT "${NPPDX_EXAMPLE_DATA_ENV}"
        )
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${NPPDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    if(NPPDX_INSTALL_EXAMPLES)
        install_example(${EXAMPLE_TARGET})
    endif()
endfunction()

get_property(NPPDX_GLOBAL_COMPILE_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
foreach(NPPDX_GLOBAL_COMPILE_DEFINITION IN LISTS NPPDX_GLOBAL_COMPILE_DEFINITIONS)
    list(APPEND NPPDX_CLANG_PTX_DEFINITION_ARGS "-D${NPPDX_GLOBAL_COMPILE_DEFINITION}")
endforeach()

# ###############################################################
# add_nppdx_clang_ptx_example
# ###############################################################
function(add_nppdx_clang_ptx_example GROUP_TARGET)
    set(EXAMPLE_SOURCES ${ARGN})
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    list(GET EXAMPLE_SOURCES 1 EXAMPLE_DEVICE_SOURCE)

    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")
    set(EXAMPLE_NAME "nppdx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")
    set(EXAMPLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    if(NOT IS_ABSOLUTE "${EXAMPLE_MAIN_SOURCE}")
        set(EXAMPLE_MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_MAIN_SOURCE}")
    endif()
    if(NOT IS_ABSOLUTE "${EXAMPLE_DEVICE_SOURCE}")
        set(EXAMPLE_DEVICE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_DEVICE_SOURCE}")
    endif()

    get_filename_component(NPPDX_CLANG_PTX_PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS)
    if(EXISTS "${NPPDX_CLANG_PTX_PROJECT_SOURCE_DIR}/libnppdx/include/nppdx.hpp")
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS
            "${NPPDX_CLANG_PTX_PROJECT_SOURCE_DIR}/libnppdx/include"
            "${CMAKE_BINARY_DIR}/libnppdx/include"
        )
    elseif(nppdx_INCLUDE_DIRS)
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS ${nppdx_INCLUDE_DIRS})
    elseif(nppdx_INCLUDE_DIR)
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS "${nppdx_INCLUDE_DIR}")
    elseif(TARGET nppdx::nppdx)
        get_target_property(NPPDX_TARGET_INCLUDE_DIRS nppdx::nppdx INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT NPPDX_TARGET_INCLUDE_DIRS)
            message(FATAL_ERROR "Unable to resolve NPPDx include directory for Clang PTX examples")
        endif()
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS ${NPPDX_TARGET_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Unable to resolve NPPDx include directory for Clang PTX examples")
    endif()

    if(nppdx_commondx_INCLUDE_DIR AND EXISTS "${nppdx_commondx_INCLUDE_DIR}/commondx/detail/config.hpp")
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS "${nppdx_commondx_INCLUDE_DIR}")
    elseif(TARGET commondx::commondx)
        get_target_property(COMMONDX_INCLUDE_DIRS commondx::commondx INTERFACE_INCLUDE_DIRECTORIES)
        list(APPEND NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS ${COMMONDX_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Unable to resolve commonDx include directory for Clang PTX examples")
    endif()

    set(NPPDX_CLANG_PTX_INCLUDE_ARGS "-I${NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR}")
    list(APPEND NPPDX_CLANG_PTX_INCLUDE_ARGS "-I${CMAKE_CURRENT_SOURCE_DIR}/common")
    foreach(NPPDX_CLANG_PTX_INCLUDE_DIR IN LISTS NPPDX_CLANG_PTX_EXAMPLE_INCLUDE_DIRS)
        if(NOT NPPDX_CLANG_PTX_INCLUDE_DIR OR "${NPPDX_CLANG_PTX_INCLUDE_DIR}" MATCHES "^\\$\<INSTALL_INTERFACE:")
            continue()
        endif()
        list(APPEND NPPDX_CLANG_PTX_INCLUDE_ARGS "-I${NPPDX_CLANG_PTX_INCLUDE_DIR}")
    endforeach()

    file(GLOB_RECURSE NPPDX_CLANG_PTX_COMPAT_HEADERS CONFIGURE_DEPENDS
        "${NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR}/*")
    file(GLOB_RECURSE NPPDX_CLANG_PTX_EXAMPLE_COMPAT_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/common/cuda_compat/*")
    list(APPEND NPPDX_CLANG_PTX_COMPAT_HEADERS ${NPPDX_CLANG_PTX_EXAMPLE_COMPAT_HEADERS})

    file(MAKE_DIRECTORY "${EXAMPLE_BINARY_DIR}")
    set(NPPDX_CLANG_PTX_OUTPUTS)
    # Compile ptxas kernels
    foreach(NPPDX_CLANG_CUDA_ARCH IN LISTS NPPDX_CUDA_ARCHITECTURES)
        nppdx_cmake_arch_to_clang_arch(NPPDX_CLANG_GPU_ARCH "${NPPDX_CLANG_CUDA_ARCH}")
        string(REGEX MATCH "^[0-9]+" NPPDX_CLANG_ARCH_NUMBER "${NPPDX_CLANG_CUDA_ARCH}")
        math(EXPR NPPDX_CLANG_DEVICE_ARCH "${NPPDX_CLANG_ARCH_NUMBER} * 10")
        set(NPPDX_CLANG_PTX_OUTPUT "${EXAMPLE_BINARY_DIR}/${EXAMPLE_TARGET}_ptx.${NPPDX_CLANG_GPU_ARCH}.ptx")
        add_custom_command(
            OUTPUT "${NPPDX_CLANG_PTX_OUTPUT}"
            COMMAND "${NPPDX_CLANG_PTX_COMPILER}"
                --cuda-device-only
                -nocudainc
                -nocudalib
                -S
                --cuda-gpu-arch=${NPPDX_CLANG_GPU_ARCH}
                -std=c++17
                -fno-exceptions
                -Wno-c++11-narrowing
                -Wno-unknown-cuda-version
                -include cuda_runtime.h
                ${NPPDX_CLANG_PTX_INCLUDE_ARGS}
                -DNPPDX_CLANG_CUDA_COMPAT
                "-DNPPDX_CLANG_DEVICE_ARCH=${NPPDX_CLANG_DEVICE_ARCH}"
                ${NPPDX_CLANG_PTX_DEFINITION_ARGS}
                "${EXAMPLE_DEVICE_SOURCE}"
                -o "${NPPDX_CLANG_PTX_OUTPUT}"
            DEPENDS "${EXAMPLE_DEVICE_SOURCE}" ${NPPDX_CLANG_PTX_COMPAT_HEADERS}
            VERBATIM
            COMMENT "Compiling ${EXAMPLE_DEVICE_SOURCE} to ${NPPDX_CLANG_GPU_ARCH} PTX with clang"
        )
        list(APPEND NPPDX_CLANG_PTX_OUTPUTS "${NPPDX_CLANG_PTX_OUTPUT}")
    endforeach()

    set(NPPDX_CLANG_PTX_TARGET "${EXAMPLE_TARGET}_ptx")
    add_custom_target(${NPPDX_CLANG_PTX_TARGET} ALL DEPENDS ${NPPDX_CLANG_PTX_OUTPUTS})

    add_executable(${EXAMPLE_TARGET} "${EXAMPLE_MAIN_SOURCE}")
    target_compile_features(${EXAMPLE_TARGET} PRIVATE cxx_std_17)
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE NPPDX_CLANG_CUDA_COMPAT)
    target_include_directories(${EXAMPLE_TARGET} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")
    if(CMAKE_DL_LIBS)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_BINARY_DIR}"
            OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )
    add_dependencies(${EXAMPLE_TARGET} ${NPPDX_CLANG_PTX_TARGET})
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})

    foreach(NPPDX_CLANG_PTX_OUTPUT IN LISTS NPPDX_CLANG_PTX_OUTPUTS)
        get_filename_component(NPPDX_CLANG_PTX_NAME "${NPPDX_CLANG_PTX_OUTPUT}" NAME_WE)
        get_filename_component(NPPDX_CLANG_PTX_FILE "${NPPDX_CLANG_PTX_OUTPUT}" NAME)
        get_filename_component(NPPDX_CLANG_PTX_DIR "${NPPDX_CLANG_PTX_OUTPUT}" DIRECTORY)
        add_test(
            NAME "${EXAMPLE_NAME}.${NPPDX_CLANG_PTX_NAME}"
            COMMAND "./$<PATH:RELATIVE_PATH,$<TARGET_FILE:${EXAMPLE_TARGET}>,${CMAKE_CURRENT_BINARY_DIR}>" "$<PATH:RELATIVE_PATH,${NPPDX_CLANG_PTX_OUTPUT},${CMAKE_CURRENT_BINARY_DIR}>"
        )
        set_tests_properties("${EXAMPLE_NAME}.${NPPDX_CLANG_PTX_NAME}"
            PROPERTIES
                LABELS "clang_ptx;EXAMPLE"
                ENVIRONMENT "${NPPDX_EXAMPLE_DATA_ENV}"
        )
    endforeach()
endfunction()

# ###############################################################
# NPPDx Examples
# ###############################################################

add_custom_target(nppdx_examples)

# NPPDx NVRTC examples
#add_nppdx_nvrtc_example(nppdx_examples 02_nvrtc/nvrtc_simple.cpp)

# NPPDx examples
if(NPPDX_BUILD_CUDA_EXAMPLES)
    add_nppdx_example(nppdx_examples 00_introduction/introduction_example.cu)
    add_nppdx_example(nppdx_examples 00_introduction/introduction_example_shared_memory.cu)
    add_nppdx_example(nppdx_examples 00_introduction/texture_example.cu)
    add_nppdx_example(nppdx_examples 01_color_conversion/image_conversion_example.cu)
    add_nppdx_example(nppdx_examples 01_color_conversion/limited_range_example.cu)
    add_nppdx_example(nppdx_examples 02_fused_pointwise/rgb_gamma_example.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/box_filter.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/box_filter_fused_conversion.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/gaussian_blur_fused_conversion.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/gaussian_blur_fused_conversion_wide.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/sharpen_fused_conversion.cu)
    add_nppdx_example_with_npp(nppdx_examples 03_area_operation/median_fused_conversion.cu)
    add_nppdx_example(nppdx_examples 04_resize/resize_example.cu)
    add_nppdx_example_with_npp(nppdx_examples 04_resize/fused_resize.cu)
endif()

if(NPPDX_BUILD_CLANG_PTX_EXAMPLES)
    add_custom_target(nppdx_clang_examples)
    add_nppdx_clang_ptx_example(nppdx_examples clang_ptx/fused_resize.cpp clang_ptx/fused_resize_device.cu)
    add_dependencies(nppdx_clang_examples clang_ptx_fused_resize)
endif()
