import meshio
msh = meshio.read("mitgcm.msh")

meshio.write("mesh.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
meshio.write("mf.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells["triangle"]},
                                    cell_data={"triangle": {"name_to_read": msh.cell_data["triangle"]["gmsh:physical"]}}))
