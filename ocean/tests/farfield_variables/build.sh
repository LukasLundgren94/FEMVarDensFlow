#!/bin/sh

cd ../../../build;
cmake ..;
make -j 1 install;
cd ../ocean/tests/farfield_variables;
cmake .;
make -j 1;
rm *vtu *pvd -r results
