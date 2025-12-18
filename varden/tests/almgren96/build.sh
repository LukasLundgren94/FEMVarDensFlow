#!/bin/sh

cd ../../../build;
cmake ..;
make -j 2 install;
cd ../varden/tests/almgren96;
cmake .;
make -j 2;
#rm *vtu *pvd -r results

