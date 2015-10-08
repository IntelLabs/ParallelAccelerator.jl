#!/bin/sh

CONF_FILE="generated/config.jl"

if [ -e "$CONF_FILE" ]
then
  rm -f "$CONF_FILE"
fi

if type "bcpp" >/dev/null 2>&1; then
    echo "use_bcpp = 1" >> "$CONF_FILE"
fi

# First check for existence of icpc; failing that, use gcc.
if type "icpc" >/dev/null 2>&1; then
    CC=icpc
    echo "backend_compiler = USE_ICC" >> "$CONF_FILE"
elif type "g++" >/dev/null 2>&1; then
    CC=g++
    echo "backend_compiler = USE_GCC" >> "$CONF_FILE"
else
    echo "You must have icpc or gcc installed to use ParallelAccelerator.";
    exit 1;
fi

echo "Using $CC to build ParallelAccelerator array runtime.";
$CC -fPIC -shared -o libj2carray.so.1.0 j2c-array.cpp
