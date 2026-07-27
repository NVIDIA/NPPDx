# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was commondx-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET commondx::commondx)
    set(commondx_VERSION "1.5.1")
    # build: NPPDX-MANUAL-BUILD

    # Targets
    include("${CMAKE_CURRENT_LIST_DIR}/commondx-targets.cmake")

    set_and_check(commondx_INCLUDE_DIR  "${PACKAGE_PREFIX_DIR}/include")
    set_and_check(commondx_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
    check_required_components(commondx)
    if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        message(STATUS "Found commondx: (Version: 1.5.1, Include dirs: ${commondx_INCLUDE_DIRS})")
    endif()
endif()
