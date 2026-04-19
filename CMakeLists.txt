cmake_minimum_required(VERSION 3.10)
project(xgtevc3 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

enable_language(Fortran)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=bounds")

# ==============================================================================
# 1. Require External Library Paths (BLAS, fmt, Google Benchmark)
# ==============================================================================

# --- BLAS ---
set(BLAS_LIBRARY_PATH "" CACHE FILEPATH "Path to the BLAS library (e.g., /usr/lib/libblas.so)")
if(NOT BLAS_LIBRARY_PATH)
    message(FATAL_ERROR "BLAS_LIBRARY_PATH is required. Specify with: -DBLAS_LIBRARY_PATH=/path/to/libblas.so")
endif()

# --- fmt ---
set(FMT_INCLUDE_DIR "" CACHE PATH "Path to the fmt include directory (containing fmt/core.h)")
set(FMT_LIBRARY_PATH "" CACHE FILEPATH "Path to the fmt library (e.g., /usr/lib/libfmt.a)")
if(NOT FMT_INCLUDE_DIR OR NOT FMT_LIBRARY_PATH)
    message(FATAL_ERROR "FMT_INCLUDE_DIR and FMT_LIBRARY_PATH are required. Specify with: -DFMT_INCLUDE_DIR=... -DFMT_LIBRARY_PATH=...")
endif()

# --- Google Benchmark ---
set(BENCHMARK_INCLUDE_DIR "" CACHE PATH "Path to the google benchmark include directory")
set(BENCHMARK_LIBRARY_PATH "" CACHE FILEPATH "Path to the google benchmark library (e.g., /usr/lib/libbenchmark.a)")
if(NOT BENCHMARK_INCLUDE_DIR OR NOT BENCHMARK_LIBRARY_PATH)
    message(FATAL_ERROR "BENCHMARK_INCLUDE_DIR and BENCHMARK_LIBRARY_PATH are required. Specify with: -DBENCHMARK_INCLUDE_DIR=... -DBENCHMARK_LIBRARY_PATH=...")
endif()


# ==============================================================================
# 2. Build the Main Library (STATIC -> libxgtevc3.a)
# ==============================================================================
file(GLOB LIB_SOURCES "src/*.cpp")

add_library(xgtevc3 STATIC ${LIB_SOURCES})

target_include_directories(xgtevc3 PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(xgtevc3 PUBLIC "${BLAS_LIBRARY_PATH}")

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
    
    target_include_directories(${TEST_NAME} PRIVATE "${FMT_INCLUDE_DIR}")
    target_link_libraries(${TEST_NAME} PRIVATE xgtevc3 "${FMT_LIBRARY_PATH}")
    
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
    
    target_include_directories(${BENCH_NAME} PRIVATE 
        "${FMT_INCLUDE_DIR}" 
        "${BENCHMARK_INCLUDE_DIR}"
    )
    
    target_link_libraries(${BENCH_NAME} PRIVATE 
        xgtevc3 
        "${FMT_LIBRARY_PATH}" 
        "${BENCHMARK_LIBRARY_PATH}"
        pthread 
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