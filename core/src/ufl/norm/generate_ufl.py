# Copyright (C) 2014 Murtazo Nazarov
# Licensed under the MIT License – see the LICENSE file for details
# 
# 
# 

cell = ['triangle', 'tetrahedron']
degree_cg = [1, 2, 3, 4]
degree_dg = [0, 0, 0, 0]

#------------------------
# Don't edit below

import sys
import os
import argparse


def main(files):
    for f in files:
        print(" ")
        print("Generating ufl files for", f)
        
        name = f.split(".")[0]
        sys.path.insert(0, os.environ["FEM_VARDEN_FLOW_BIN"])
        import ufl_generate
        ufl_generate.generate(cell, degree_cg, degree_dg, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+', help='path to the file')
    args_namespace = parser.parse_args()
    args = vars(args_namespace)['file']
    main(args)
