#!/bin/sh

cd ../../../build;
cmake ..;
make -j 2 install;
cd ../varden/tests/buoyancy-driven-cavity;
cmake .;
make -j 2;
rm *vtu *pvd -r results
