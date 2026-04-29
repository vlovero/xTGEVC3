cmake_minimum_required(VERSION 3.12)
project(fxgtevc3 VERSION 0.1.0 LANGUAGES Fortran)

include(GNUInstallDirs)

# ==============================================================================
# 1. Build the Fortran Library (STATIC -> libfxgtevc3.a)
# ==============================================================================
file(GLOB LIB_SOURCES "src/*.f")

add_library(fxgtevc3 STATIC ${LIB_SOURCES})

# Inherits the BLAS package found in the parent directory
target_link_libraries(fxgtevc3 PUBLIC ${BLAS_LIBRARIES})


# ==============================================================================
# 2. Create Test Executables
# ==============================================================================
enable_testing()

file(GLOB TEST_SOURCES "tests/*.f")
set(TEST_TARGETS "") # Keep track of the test targets for installation

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    set(FULL_TEST_NAME "f${TEST_NAME}")
    
    add_executable(${FULL_TEST_NAME} ${TEST_SRC})
    
    # Replaced the hardcoded BLAS_LIBRARY_PATH with the modern BLAS_LIBRARIES variable
    target_link_libraries(${FULL_TEST_NAME} PRIVATE 
        fxgtevc3 
        ${BLAS_LIBRARIES}
    )
    
    add_test(NAME ${FULL_TEST_NAME} COMMAND ${FULL_TEST_NAME})
    list(APPEND TEST_TARGETS ${FULL_TEST_NAME})
endforeach()


# ==============================================================================
# 3. Install Rules
# ==============================================================================

# Install the library binary
install(TARGETS fxgtevc3
    EXPORT fxgtevc3Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install the test executables
if(TEST_TARGETS)
    install(TARGETS ${TEST_TARGETS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/fortran
    )
endif()