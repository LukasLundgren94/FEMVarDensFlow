# All 
cell = ['triangle','tetrahedron']
degree_V_M = [3,2,1,3,3,4]
degree_Q   = [2,1,1,3,2,3]


# Only p2p1 in 2D
#cell = ['triangle']
#degree_V_M = [2]
#degree_Q = [1]

#------------------------
# Don't edit below

import sys
import os
import argparse

# Generate the Equation file
#------------------------------------------------------------------
def generate_h(cell, degree_V_M, degree_Q, name):

    ufl_src_file = open(name + '.ufl.src', 'r')
    ufc_src = ufl_src_file.read()

    dim = []
    for c in cell:
        if c == 'interval':
            dim.append(1)
        if c == 'triangle':
            dim.append(2)
        if c == 'tetrahedron':
            dim.append(3)
            
    file_name = name+"Equation.h"
    print("Generating the Equation header file: ", file_name)
    file = open(file_name, 'w+')
    file.write("// Generated automatically\n")
    file.write("\n")
    file.write("\n")
    file.write("#ifndef __" + name + "EQUATION_H\n")
    file.write("#define __" + name + "EQUATION_H\n")
    file.write("\n")
    file.write("\n")
    file.write("#include <dolfin/fem/Form.h>\n")
    file.write("#include <dolfin/mesh/Mesh.h>\n")
    file.write("\n")

    for d in range(0, len(dim)):

        for p in range(0, len(degree_V_M)):
            file.write("#include \"" + "ufl/"+ name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) + ".h\"\n")

            print("#include \"" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) + ".h\"")


    file.write("\n")
    file.write("using namespace dolfin;\n")
    file.write("\n")
    file.write("namespace fem_varden_flow\n")
    file.write("{\n")
    file.write("class " + name + "Equation : public Variable\n")
    file.write("{\n")
    file.write("public:\n")

    file.write("\n")
    file.write("//-----------------------------------------------------------------------------\n")
    file.write("static void init(std::shared_ptr<const Mesh> mesh, System &eq0, System &eq1, System &eq2, System &eq3, System &eqf, System &eqS, Parameters &p)\n")
    file.write("{\n")
    file.write("  const std::size_t D = mesh->topology().dim();\n")
    file.write("  int const pol_degree_V_M = p[\"PolDegree_V_M\"];\n")
    file.write("  int const pol_degree_Q = p[\"PolDegree_Q\"];\n")
    file.write("  ///\n\n")

    for d in range(0, len(dim)):
        file.write("  if (D == " + str(dim[d]) + ")\n")
        file.write("  {\n") 

        for p in range(0, len(degree_V_M)):
            file.write("    if ( ( pol_degree_V_M == " +str(degree_V_M[p]) + ") && ( pol_degree_Q == " +str(degree_Q[p]) + ") ) \n") 
            file.write("      {\n") 
            file.write("        if (!eq0.V) \n") 
            file.write("          eq0.V = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::FunctionSpace>(mesh); \n") 
            file.write("       eq0.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_a0_s>(eq0.V, eq0.V); \n") 
            file.write("       eq0.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_L0_s>(eq0.V); \n") 
            file.write("       eq1.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_a1_s>(eq0.V, eq0.V); \n") 
            file.write("       eq1.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_L1_s>(eq0.V); \n") 
            file.write("       eq2.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_a2_s>(eq0.V, eq0.V); \n") 
            file.write("       eq2.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_L2_s>(eq0.V); \n") 
            file.write("       eq3.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_a3_s>(eq0.V, eq0.V); \n") 
            file.write("       eq3.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_L3_s>(eq0.V); \n")
            file.write("       eqf.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_af_s>(eq0.V, eq0.V); \n") 
            file.write("       eqf.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_Lf_s>(eq0.V); \n")
            file.write("       eqs.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_aSchur>(eq0.V, eq0.V); \n") 
            # file.write("       eqi.a = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_ai>(eq0.V, eq0.V); \n") 
            # file.write("       eqi.L = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::Form_Li>(eq0.V); \n")
            file.write("       eq0.DG0 = std::make_shared<" + name + "_" + str(dim[d]) + "D_P" + str(degree_V_M[p]) + "P" + str(degree_Q[p]) +"::CoefficientSpace_Norm_rho>(mesh); \n")


            file.write("      }\n") 
            
        file.write("  }\n\n")
    file.write("}\n")
    file.write("  ///\n")
        
    file.write("};\n")
    file.write("}\n\n")
    file.write("#endif\n")
    file.close()


#------------------------------------------------------------------
def main(files):
    for f in files:
        print(" ")
        print("Generating ufl files for", f)
        
        name = f.split(".")[0]
        sys.path.insert(0, os.environ["FEM_VARDEN_FLOW_BIN"])
        import ufl_generate_VD
        ufl_generate_VD.generate(cell, degree_V_M, degree_Q, name)

        # Generate the Equation header file
        generate_h(cell, degree_V_M, degree_Q, name)

#------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+', help='path to the file')
    args_namespace = parser.parse_args()
    args = vars(args_namespace)['file']
    main(args)


