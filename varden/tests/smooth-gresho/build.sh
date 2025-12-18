#!/bin/sh

cd ../../../build;
cmake ..;
make -j 4 install;
cd ../varden/tests/smooth-gresho;
cmake .;
make -j 4;
rm *vtu *pvd -r results

