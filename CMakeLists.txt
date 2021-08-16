
cmake_minimum_required(VERSION 3.3.0)
project(intel)

# default ABI(application binary interface) is x86-64
if (ABI)
    message("target arch is ${ABI}")
else()
    set(ABI x86-64)
    message("target arch is x86-64")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic  -fPIC -ldl  -lm")

# *.cc files
file(GLOB SRCS
    src/*.cpp
    src/operator/*.cpp
    )

#if (BUILD_OPENMVG)
#    set(SRCS
#        ${SRCS}
#        ${CMAKE_CURRENT_SOURCE_DIR}/src/face3d/face3d.cc)
#endif()

add_library(intel STATIC ${SRCS})

if (${ABI} STREQUAL "armeabi-v7a")
    target_compile_definitions(intel PUBLIC __ANDROID__)
    target_compile_options(intel PUBLIC -mfloat-abi=softfp -mfpu=neon)
    find_library(log-lib log)
    target_link_libraries(intel PUBLIC ${log-lib})
endif()
string(REGEX REPLACE ".*/\(.*\)" "\\1" CURDIR ${CMAKE_CURRENT_SOURCE_DIR})
message("** dir: " ${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(/Users/huifeng/fh/myproject/include/intel)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/intel)


# set compile flag
target_compile_options(intel PUBLIC -frtti -fexceptions)
if (BUILD_SONIC_DEBUG)
    target_compile_options(intel PUBLIC -g -O0)
else()
    target_compile_options(intel PUBLIC -O3)
endif()

target_compile_options(intel PUBLIC -std=c++14 -Wpedantic -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-unused-variable)

set_target_properties(intel PROPERTIES POSITION_INDEPENDENT_CODE True)

target_link_libraries(
    intel
    PUBLIC
    )
add_subdirectory(tests)

if (BUILD_TOOLS)
   add_subdirectory(tools)
endif()
message("\n")
message("** arch: " ${ABI})
message("\n")

install(TARGETS intel ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX})
