cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)

project(div_ascendc LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(div_ascendc src/div_host.asc)

target_link_libraries(div_ascendc PRIVATE
    ascendcl
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
)

target_include_directories(div_ascendc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ACL_INCLUDE_DIR}
)

target_compile_options(div_ascendc PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

target_link_directories(div_ascendc PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
)
