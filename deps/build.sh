#!/bin/bash

# Copyright (c) 2015, Intel Corporation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice, 
#   this list of conditions and the following disclaimer in the documentation 
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
# THE POSSIBILITY OF SUCH DAMAGE.

# This script is run by build.jl when one runs
# `Pkg.build("ParallelAccelerator")` at the Julia REPL.  Among other
# things, it tries to determine what C++ compiler the user has,
# whether that compiler has OpenMP support, and what BLAS library, if
# any, is available.

CONF_FILE="generated/config.jl"
MKL_LIB=""
OPENBLAS_LIB=""
OPENMP_SUPPORTED=""

if [ -e "$CONF_FILE" ]
then
  rm -f "$CONF_FILE"
fi

# C++ compiler checks.  We first check for existence of icpc; failing
# that, we check for g++.
if type "icpc" >/dev/null 2>&1; then
    CC=icpc
    echo "backend_compiler = USE_ICC" >> "$CONF_FILE"
elif type "g++" >/dev/null 2>&1; then
    CC=g++
    echo "backend_compiler = USE_GCC" >> "$CONF_FILE"
else
    echo "You won't be able to use the cgen backend because you don't have icpc or g++ installed.";
    echo "use_bcpp = 0" >> "$CONF_FILE"
    echo "backend_compiler = NONE" >> "$CONF_FILE"
    echo "mkl_lib = \"\"" >> "$CONF_FILE"
    echo "openblas_lib = \"\"" >> "$CONF_FILE"
    echo "sys_blas = 0" >> "$CONF_FILE"
    echo "openmp_supported = 0" >> "$CONF_FILE"
    exit 0;
fi

# Check for presence of bcpp, a C++ code formatting tool.  This is
# completely optional; nothing will break if bcpp is not present, but
# it makes the output of the ParallelAccelerator compiler easier to
# read.
if type "bcpp" >/dev/null 2>&1; then
    echo "use_bcpp = 1" >> "$CONF_FILE"
fi

# When build.jl runs, it passes the contents of the DYLD_LIBRARY_PATH
# and LD_LIBRARY_PATH environment variables to this script.  This
# script parses them and checks for the presence of MKL and OpenBLAS
# shared libraries.
syslibs=${BASH_ARGV[*]}
arr_libs=(${syslibs//:/ })

for lib in "${arr_libs[@]}"
do
    # We first check for MKL; failing that, we check for OpenBLAS.
    if echo "$lib" | grep -q "/mkl/"; then
        MKL_LIB=$lib
    elif echo "$lib" | grep -q "OpenBLAS\|openblas"; then
        OPENBLAS_LIB=$lib
    fi
done

# After running the above, if MKL_LIB is still not set, try compiling
# a simple test program that uses it.  If it works, assume the MKL
# shared library is in `/opt/intel/mkl/lib`.
if [ -z "$MKL_LIB" ]; then
    echo "#include <mkl.h>" > blas_test.cpp
    echo "int main(){return 0;}" >> blas_test.cpp
    CHECK_MKL_COMPILE=`$CC blas_test.cpp -mkl 2>&1`
    rm blas_test.cpp
    # If the simple test program compiles with no errors, we'll assume
    # MKL is present and working.
    if [ -z "$CHECK_MKL_COMPILE" ]; then
        echo "System installed MKL found"
        MKL_LIB="/opt/intel/mkl/lib"
    fi
fi

# Check whether a standard `cblas.h` header is available, and whether
# a simple test program that uses it compiles with the `-lblas` flag.
echo "#include <cblas.h>" > blas_test.cpp
echo "int main(){return 0;}" >> blas_test.cpp
BLAS_COMPILE=`$CC blas_test.cpp -lblas 2>&1`
rm blas_test.cpp
# If the simple test program compiles with no errors, we'll assume a
# system BLAS library is present and working.
if [ -z "$BLAS_COMPILE" ]; then
    echo "System installed BLAS found"
    SYS_BLAS=1
else
    SYS_BLAS=0
fi

# Check whether a standard `cblas.h` header is available, and whether
# a simple test program that uses it compiles with the `-lopenblas`
# flag.
echo "#include <cblas.h>" > blas_test.cpp
echo "int main(){return 0;}" >> blas_test.cpp
OPENBLAS_COMPILE=`$CC blas_test.cpp -lopenblas 2>&1`
rm blas_test.cpp
# If the simple test program compiles with no errors, we'll assume
# OpenBLAS is present and working.
if [ -z "$OPENBLAS_COMPILE" ]; then
    echo "OpenBLAS found"
    # At this point, hopefully OPENBLAS_LIB already points to the
    # correct directory for the shared library if needed.
else
    unset OPENBLAS_LIB
fi

#echo "out" $SYS_BLAS $MKL_LIB $OPENBLAS_LIB

# Check whether the C++ compiler supports OpenMP or not.
echo "Checking for OpenMP support..."
echo "#include <omp.h>" > openmp_test.cpp
echo "#include <stdio.h>" >> openmp_test.cpp
echo "int main() { printf(\"Max OpenMP threads: %d\n\", omp_get_max_threads()); }"  >> openmp_test.cpp
OPENMP_COMPILE=`$CC openmp_test.cpp -fopenmp -o openmp_test 2>&1`
rm openmp_test.cpp

if [ -z "$OPENMP_COMPILE" ]; then
    echo "OpenMP support found in $CC"
    OPENMP_SUPPORTED=1
    OPENMP_RUN=`./openmp_test`
    echo $OPENMP_RUN
    rm openmp_test
else
    echo "No OpenMP support found in $CC"
    OPENMP_SUPPORTED=0
fi

# If neither MKL_LIB nor OPENBLAS_LIB are set and SYS_BLAS is 0, then
# none of our attempts to find a BLAS library worked and we will
# proceed assuming that there isn't one.
if [ -z "$MKL_LIB" ] && [ -z "$OPENBLAS_LIB" ] && [ "$SYS_BLAS" -eq "0" ]; then
    echo "No BLAS installation detected (optional)"
fi

if [ -n "$NERSC_HOST" ]; then
    echo "Configuring for LBL NERSC machines"
     echo "NERSC = 1" >> "$CONF_FILE"
fi

echo "mkl_lib = \"$MKL_LIB\"" >> "$CONF_FILE"
echo "openblas_lib = \"$OPENBLAS_LIB\"" >> "$CONF_FILE"
echo "sys_blas = $SYS_BLAS" >> "$CONF_FILE"
echo "openmp_supported = $OPENMP_SUPPORTED" >> "$CONF_FILE"

echo "Using $CC to build ParallelAccelerator array runtime.";
$CC -std=c++11 -fPIC -shared -o libj2carray.so.1.0 j2c-array.cpp
