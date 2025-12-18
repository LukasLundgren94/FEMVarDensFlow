#!/bin/sh

cd ../../../build;
cmake ..;
make -j 4 install;
cd ../varden/tests/lock-exchange-3D;
cmake .;
make -j 4;
rm *vtu *pvd -r results
