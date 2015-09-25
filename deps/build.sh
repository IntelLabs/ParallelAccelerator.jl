# First check for existence of icpc; failing that, use gcc.
if type "icpc"; then
    CC=icpc
elif type "gcc"; then
    CC=gcc
else
    echo "You must have icpc or gcc installed to use ParallelAccelerator.";
    exit 1;
fi

echo "Using $CC to build ParallelAccelerator array runtime.";
$CC -fPIC -shared -o libj2carray.so.1.0 j2c-array.cpp
