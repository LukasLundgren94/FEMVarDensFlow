#!/bin/sh

cd ../../../build;
cmake ..;
make -j 1 install;
cd ../varden/tests/well-balanced;
cmake .;
make -j 1;
rm *vtu *pvd -r results
