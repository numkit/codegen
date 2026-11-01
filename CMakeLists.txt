cmake_minimum_required(VERSION 3.21)
project(numkit_codegen VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/MP /bigobj)
else()
    add_compile_options(-Wall -Wextra -fPIC)
endif()

option(NUMKIT_WITH_SIMD   "Enable Highway SIMD acceleration" ON)
option(NUMKIT_BUILD_CLI   "Build numkit_codegen command-line executable" ON)
option(NUMKIT_BUILD_TESTS "Build GoogleTest suite for numkit_codegen" ON)

include(FetchContent)

# ── Dependency: numkit (and transitive value / sys / ops) ────────────────
if(NOT TARGET numkit)
    set(_nk_prebuilt_dir "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build")
    find_library(_nk_lib_numkit NAMES numkit PATHS "${_nk_prebuilt_dir}" "${_nk_prebuilt_dir}/Release" NO_DEFAULT_PATH)
    find_library(_nk_lib_ops    NAMES numkit_ops PATHS "${_nk_prebuilt_dir}/_deps/numkit_ops" "${_nk_prebuilt_dir}/_deps/numkit_ops/Release" NO_DEFAULT_PATH)
    find_library(_nk_lib_sys    NAMES numkit_sys PATHS "${_nk_prebuilt_dir}/_deps/numkit_sys" "${_nk_prebuilt_dir}/_deps/numkit_sys/Release" NO_DEFAULT_PATH)
    find_library(_nk_lib_value  NAMES numkit_value PATHS "${_nk_prebuilt_dir}/_deps/numkit_value" "${_nk_prebuilt_dir}/_deps/numkit_value/Release" NO_DEFAULT_PATH)
    find_library(_nk_lib_hwy    NAMES hwy PATHS "${_nk_prebuilt_dir}/_deps/highway-build" "${_nk_prebuilt_dir}/_deps/highway-build/Release" NO_DEFAULT_PATH)
    find_library(_nk_lib_hwy_c  NAMES hwy_contrib PATHS "${_nk_prebuilt_dir}/_deps/highway-build" "${_nk_prebuilt_dir}/_deps/highway-build/Release" NO_DEFAULT_PATH)
    if(NOT NUMKIT_BUILD_TESTS AND _nk_lib_numkit AND _nk_lib_ops AND _nk_lib_sys AND _nk_lib_value)
        message(STATUS "Using prebuilt numkit libraries from ${_nk_prebuilt_dir}")
        add_library(numkit INTERFACE)
        add_library(numkit::numkit ALIAS numkit)
        target_include_directories(numkit INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}/../value/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../sys/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../ops/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/include"
        )
        target_link_libraries(numkit INTERFACE
            "${_nk_lib_numkit}"
            "${_nk_lib_ops}"
            "${_nk_lib_sys}"
            "${_nk_lib_value}"
        )
        if(_nk_lib_hwy AND _nk_lib_hwy_c)
            target_link_libraries(numkit INTERFACE "${_nk_lib_hwy}" "${_nk_lib_hwy_c}")
        endif()
        find_package(Threads REQUIRED)
        target_link_libraries(numkit INTERFACE Threads::Threads)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/CMakeLists.txt")
        message(STATUS "Using local numkit/numkit from ${CMAKE_CURRENT_SOURCE_DIR}/../numkit")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../numkit" _deps/numkit)
    else()
        FetchContent_Declare(
            numkit
            GIT_REPOSITORY https://github.com/numkit/numkit.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit)
    endif()
endif()

# ── AOT Compiler Configuration ───────────────────────────────────────────
set(NUMKIT_CODEGEN_AOT_AVAILABLE 0)
set(NUMKIT_CODEGEN_AOT_IS_MSVC 0)
set(NUMKIT_CODEGEN_AOT_VCVARS "")
set(NUMKIT_CODEGEN_AOT_CXX "${CMAKE_CXX_COMPILER}")
if(MSVC AND CMAKE_GENERATOR_INSTANCE)
    set(_nk_vcvars "${CMAKE_GENERATOR_INSTANCE}/VC/Auxiliary/Build/vcvars64.bat")
    if(EXISTS "${_nk_vcvars}")
        set(NUMKIT_CODEGEN_AOT_AVAILABLE 1)
        set(NUMKIT_CODEGEN_AOT_IS_MSVC 1)
        set(NUMKIT_CODEGEN_AOT_VCVARS "${_nk_vcvars}")
    endif()
elseif(NOT MSVC AND CMAKE_CXX_COMPILER)
    set(NUMKIT_CODEGEN_AOT_AVAILABLE 1)
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/aot_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/numkit/codegen/aot_config.hpp
    @ONLY
)

# ── Library: numkit_codegen_core ─────────────────────────────────────────
file(GLOB_RECURSE NUMKIT_CODEGEN_SOURCES
    "src/*.cpp"
)

add_library(numkit_codegen_core STATIC ${NUMKIT_CODEGEN_SOURCES})
add_library(numkit::codegen ALIAS numkit_codegen_core)

target_include_directories(numkit_codegen_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bridge>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_codegen_core
    PUBLIC
        numkit
)

# ── Executable: numkit_codegen CLI ───────────────────────────────────────
if(NUMKIT_BUILD_CLI AND NOT EMSCRIPTEN)
    add_executable(numkit_codegen apps/numkit_codegen/main.cpp)
    target_link_libraries(numkit_codegen PRIVATE numkit_codegen_core)
endif()

# ── Shared Runtime Libraries & GoogleTest Suite ──────────────────────────
if(NUMKIT_BUILD_TESTS AND NOT EMSCRIPTEN)
    enable_testing()

    if(NOT TARGET GTest::gtest_main)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # 1. Sample standalone plugin (DESIGN.md §6b)
    add_library(numkit_sample_plugin SHARED ${CMAKE_CURRENT_SOURCE_DIR}/tests/sample_plugin.cpp)
    target_include_directories(numkit_sample_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bridge)
    set_target_properties(numkit_sample_plugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
    )

    set(NUMKIT_SAMPLE_PLUGIN_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}numkit_sample_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}")
    get_filename_component(NUMKIT_BRIDGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bridge" ABSOLUTE)

    # 2. Value-ABI bridge runtime as standalone shared library (DESIGN.md §6a)
    add_library(numkit_codegen_rt_shared SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bridge/numkit_codegen_rt.cpp)
    target_include_directories(numkit_codegen_rt_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bridge)
    target_compile_definitions(numkit_codegen_rt_shared PRIVATE NUMKIT_RT_BUILDING_DLL)
    target_link_libraries(numkit_codegen_rt_shared PRIVATE numkit)
    set_target_properties(numkit_codegen_rt_shared PROPERTIES
        OUTPUT_NAME                      numkit_codegen_rt
        RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
    )
    set(NUMKIT_RT_IMPORT_LIB
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_IMPORT_LIBRARY_PREFIX}numkit_codegen_rt${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    set(NUMKIT_RT_SHARED_DLL
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}numkit_codegen_rt${CMAKE_SHARED_LIBRARY_SUFFIX}")

    # 3. Public ops kernel facade as standalone shared library
    get_target_property(_numkit_ops_src_dir numkit_ops SOURCE_DIR)
    add_library(numkit_ops_kernels SHARED "${_numkit_ops_src_dir}/src/kernels/kernels.cpp")
    target_include_directories(numkit_ops_kernels PRIVATE
        "${_numkit_ops_src_dir}/include"
        "${_numkit_ops_src_dir}/src"
    )
    target_compile_definitions(numkit_ops_kernels PRIVATE NUMKIT_OPS_BUILDING_DLL)
    target_link_libraries(numkit_ops_kernels PRIVATE numkit_ops)
    set_target_properties(numkit_ops_kernels PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY         ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
    )
    get_filename_component(NUMKIT_OPS_INCLUDE_DIR "${_numkit_ops_src_dir}/include" ABSOLUTE)
    set(NUMKIT_OPS_IMPORT_LIB
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_IMPORT_LIBRARY_PREFIX}numkit_ops_kernels${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    set(NUMKIT_OPS_SHARED_DLL
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}numkit_ops_kernels${CMAKE_SHARED_LIBRARY_SUFFIX}")

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/numkit_codegen_test_paths.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/numkit_codegen_test_paths.h
        @ONLY
    )

    set(NUMKIT_CODEGEN_TEST_SOURCES
        tests/type_lattice_test.cpp
        tests/transfer_test.cpp
        tests/transfer_diff_test.cpp
        tests/inference_test.cpp
        tests/indexing_test.cpp
        tests/emitter_test.cpp
        tests/aot_test.cpp
        tests/e2e_test.cpp
        tests/monomorphize_test.cpp
        tests/classinfo_test.cpp
        tests/driver_test.cpp
        tests/bridge_test.cpp
        tests/tiered_test.cpp
        bridge/numkit_codegen_rt.cpp
    )

    add_executable(numkit_codegen_test ${NUMKIT_CODEGEN_TEST_SOURCES})
    target_include_directories(numkit_codegen_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bridge
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_link_libraries(numkit_codegen_test PRIVATE
        numkit_codegen_core
        numkit
        GTest::gtest_main
    )
    add_dependencies(numkit_codegen_test
        numkit_sample_plugin
        numkit_codegen_rt_shared
        numkit_ops_kernels
    )
    if(TARGET numkit_stage_plugins)
        add_dependencies(numkit_codegen_test numkit_stage_plugins)
    endif()
    add_test(NAME numkit_codegen_test COMMAND numkit_codegen_test)
endif()
