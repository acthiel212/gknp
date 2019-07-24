#---------------------------------------------------
# OpenMM GKNP Plugin
#----------------------------------------------------
set(CMAKE_VERBOSE_MAKEFILE on)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET (CMAKE_CXX_STANDARD 11)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/Users/andthiel/Programs/build_openmm_clean")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

# set flags for linking on mac
IF(APPLE)
    SET (CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/Users/andthiel/Programs/build_openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

#build shared GaussVol library
SET(GAUSSVOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gaussvol")
SET(GAUSSVOLLIB_NAME gaussvol)
SET(GAUSSVOLLIB_INCLUDE_DIR ${GAUSSVOLLIB_DIR})
ADD_LIBRARY(${GAUSSVOLLIB_NAME} SHARED "${GAUSSVOL_DIR}/gaussvol.cpp" "${GAUSSVOL_DIR}/gaussvol.h")
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${GAUSSVOLLIB_NAME})

#
#build/Install the main API library
#
SET(GKNP_PLUGIN_SOURCE_SUBDIRS openmmapi)
SET(GKNP_LIBRARY_NAME GKNPPlugin)
SET(SHARED_GKNP_TARGET ${GKNP_LIBRARY_NAME})
#These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "gaussvol" "openmmapi/include" "openmmapi/include/internal")
# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)
# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${GKNP_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)
INCLUDE_DIRECTORIES(BEFORE ${GAUSSVOL_DIR})

# Create the library.
ADD_LIBRARY(${SHARED_GKNP_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_GKNP_TARGET}
    PROPERTIES COMPILE_FLAGS "-DGKNP_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_GKNP_TARGET} OpenMM)
INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_GKNP_TARGET})
# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

#
# Build/Install for the Reference platform
#
ADD_SUBDIRECTORY(platforms/reference)

#to find FindOpenCL.cmake etc.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#
# Build/Install for the Cuda platform (incomplete)
#
FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    SET(GKNP_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for Cuda")
ELSE(CUDA_FOUND)
    SET(GKNP_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for Cuda")
ENDIF(CUDA_FOUND)
IF(GKNP_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(GKNP_BUILD_CUDA_LIB)

#
# Testing
#

ENABLE_TESTING()

IF (EXECUTABLE_OUTPUT_PATH)
  SET (TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
ELSE (EXECUTABLE_OUTPUT_PATH)
  SET (TEST_PATH .)
ENDIF (EXECUTABLE_OUTPUT_PATH)

IF(BUILD_TESTING AND OPENMM_BUILD_SERIALIZATION_TESTS)
    ADD_SUBDIRECTORY(serialization/tests)
ENDIF(BUILD_TESTING AND OPENMM_BUILD_SERIALIZATION_TESTS)

