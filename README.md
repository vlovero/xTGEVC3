# xgtevc3

A C++ library implementing a new blocked variant of the xTGEVC routines (xTGEVC3) for computing the eigenvectors in the generalized eigenvalue problem. 

## Requirements

* C++20 compliant compiler
* Fortran compiler/runtimes
* BLAS/LAPACK libraries (tested with OpenBLAS)
* fmt (for tests)
* Google Benchmark (for benchmarks)

## Building and Installation

The project uses CMake for building and testing. You must explicitly provide the paths to your BLAS, fmt, and Google Benchmark installations during the configuration step.

Example build script:
```bash
mkdir build
cd build

cmake -DBLAS_LIBRARY_PATH=/path/to/libopenblas.a \
      -DFMT_INCLUDE_DIR=/path/to/fmt/include \
      -DFMT_LIBRARY_PATH=/path/to/fmt/libfmt.a \
      -DBENCHMARK_INCLUDE_DIR=/path/to/googlebenchmark/include \
      -DBENCHMARK_LIBRARY_PATH=/path/to/googlebenchmark/libbenchmark.a \
      -DCMAKE_INSTALL_PREFIX=`pwd` -DCMAKE_BUILD_TYPE=Release \
      ..

cmake --build . --target install
```