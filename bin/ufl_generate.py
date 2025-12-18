# Copyright (C) 2014 Murtazo Nazarov
# 
# 
# 
# 

def generate(cell, degree_cg, degree_dg, name):
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
        for p in range(0, len(degree_cg)):
            file_name = name+"_" + str(dim[d]) + "D_P" + str(degree_cg[p]) + ".ufl"
            print(file_name)

            file = open(file_name, 'w+')
            file.write("# Generated automatically\n")
            file.write("\n")
            file.write("cell = "+cell[d]+"\n")
            file.write("degree_cg = "+str(degree_cg[p]))
            file.write("\n")
            file.write("degree_dg = "+str(degree_dg[p]))
            file.write("\n")

            file.write(ufc_src)
            file.close()
