#!/bin/sh

orig_dir=$(pwd)

cd ../../../build;
cmake  ..;
make -j 1 install;
cd "$orig_dir";
cmake  .;
make -j 1;
#rm *vtu *pvd -r results
