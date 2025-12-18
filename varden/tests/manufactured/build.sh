#!/bin/sh

cd ../../../build;
cmake ..;
make -j 1 install;
cd ../varden/tests/manufactured;
cmake .;
make -j 1;
rm *vtu *pvd -r results

