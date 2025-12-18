import gmsh

# Initialize gmsh and set up the factory
gmsh.initialize()
gmsh.model.add("square")
gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)

# Synchronize to transfer the built entities to the gmsh model
gmsh.model.occ.synchronize()

gmsh.model.geo.addLine(1,2)
gmsh.model.geo.addLine(2,3)
gmsh.model.geo.addLine(3,4)
gmsh.model.geo.addLine(4,1)

loop = gmsh.model.geo.addCurveLoop([1, 2, 3, 4])
gmsh.model.geo.addPlaneSurface([loop])

# Mesh settings
res = 10.0
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 / res)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.0 / res)
gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay triangulation for 2D

# Generate the mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Save the mesh
gmsh.write("square.msh")

# Finalize the gmsh session
gmsh.finalize()