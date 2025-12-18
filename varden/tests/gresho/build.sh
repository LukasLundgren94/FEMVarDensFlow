#!/bin/sh

cd ../../../build;
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..;
make -j 1 install;
cd ../varden/tests/gresho;
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .;
make -j 1;
rm *vtu *pvd -r results

