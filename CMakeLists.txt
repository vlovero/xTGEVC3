# Bumped minimum version to 3.12 to natively support <PackageName>_ROOT variables
cmake_minimum_required(VERSION 3.12)
project(xgtevc3 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

enable_language(Fortran)

# ==============================================================================
# 1. Require External Libraries (BLAS, fmt, Google Benchmark)
# ==============================================================================
# Users can manually specify paths during configuration using:
# -Dfmt_ROOT=/path/to/fmt 
# -Dbenchmark_ROOT=/path/to/benchmark
# -DBLA_VENDOR=... (this sometimes also need -DBLAS_LIBRARIES=...)

find_package(BLAS REQUIRED)
find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(fortran)

# ==============================================================================
# 2. Build the Main Library
# ==============================================================================
file(GLOB LIB_SOURCES "src/*.cpp" "src/*.c" "src/*.f")

add_library(xgtevc3 STATIC ${LIB_SOURCES})

target_include_directories(xgtevc3 PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Use the variables provided by FindBLAS 
target_link_libraries(xgtevc3 PUBLIC ${BLAS_LIBRARIES})

target_link_directories(xgtevc3 PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(xgtevc3 PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

# ==============================================================================
# 3. Create Test Executables
# ==============================================================================
enable_testing()

file(GLOB TEST_SOURCES "tests/*.cpp")
set(TEST_TARGETS "") # Keep track of the test targets for installation 

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE) 
    add_executable(${TEST_NAME} ${TEST_SRC}) 
    
    # Modern CMake targets handle include directories automatically
    target_link_libraries(${TEST_NAME} PRIVATE xgtevc3 fmt::fmt)
    
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}) 
    list(APPEND TEST_TARGETS ${TEST_NAME}) 
endforeach()


# ==============================================================================
# 4. Create Benchmark Executables
# ==============================================================================
file(GLOB BENCHMARK_SOURCES "benchmarks/*.cpp") 
set(BENCHMARK_TARGETS "") # Keep track of the benchmark targets for installation 

foreach(BENCH_SRC ${BENCHMARK_SOURCES}) 
    get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE) 
    add_executable(${BENCH_NAME} ${BENCH_SRC}) 
    
    # Target linkage automatically applies include paths and definitions
    target_link_libraries(${BENCH_NAME} PRIVATE 
        xgtevc3 
        fmt::fmt 
        benchmark::benchmark
        Threads::Threads
    )
    
    list(APPEND BENCHMARK_TARGETS ${BENCH_NAME}) 
endforeach()


# ==============================================================================
# 5. Install Rules
# ==============================================================================

# Install the library binary and headers
install(TARGETS xgtevc3
    EXPORT xgtevc3Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} 
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" 
)

# Install the test executables
if(TEST_TARGETS) 
    install(TARGETS ${TEST_TARGETS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
    )
endif()

# Install the benchmark executables
if(BENCHMARK_TARGETS) 
    install(TARGETS ${BENCHMARK_TARGETS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
    )
endif()
