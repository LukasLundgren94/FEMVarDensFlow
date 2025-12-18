# All 
cell = ['triangle','tetrahedron']
degree_V_M = [2,2,1,3,3,4]
degree_Q = [1,2,1,3,2,3]

# Only p2p1 in 2D
#cell = ['triangle']
#degree_V_M = [2]
#degree_Q = [1]

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
        import ufl_generate_VD
        ufl_generate_VD.generate(cell, degree_V_M, degree_Q, name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+', help='path to the file')
    args_namespace = parser.parse_args()
    args = vars(args_namespace)['file']
    main(args)
