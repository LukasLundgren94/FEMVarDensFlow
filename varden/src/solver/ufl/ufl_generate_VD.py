# Copyright (C) 2014 Murtazo Nazarov
# 
# 
# 
# 

def generate(cell, degree_V_M, degree_Q, name):
    ufl_src_file = open(name+'.ufl.src', 'r')
    ufc_src = ufl_src_file.read()

    dim = []
    for c in cell:
        if c == 'interval':
            dim.append(1)
        if c == 'triangle':
            dim.append(2)
        if c == 'tetrahedron':
            dim.append(3)
        
    for d in range(0, len(dim)):
        for p in range(0, len(degree_V_M)):
            file_name = name+"_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p])  + ".ufl"
            print(file_name)

            file = open(file_name, 'w+')
            file.write("# Generated automatically\n")
            file.write("\n")
            file.write("cell = "+cell[d]+"\n")
            file.write("degree_V_M = "+str(degree_V_M[p]))
            file.write("\n")
            file.write("degree_Q = "+str(degree_Q[p]))
            file.write("\n")

            file.write(ufc_src)
            file.close()