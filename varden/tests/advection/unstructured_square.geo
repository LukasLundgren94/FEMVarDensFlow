SetFactory("OpenCASCADE");


// Define the points of the square
Point(1) = {-1, -1, 0, 0.0001};
Point(2) = {1, -1, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {-1, 1, 0, 0.1};

// Define the lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Create a line loop and plane surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Set the mesh resolution
res = DefineNumber[256.0];
#Mesh.CharacteristicLengthMin = 0.5/res;
#Mesh.CharacteristicLengthMax = 1.0/res;

Mesh.Algorithm = 5; // Delaunay triangulation for 2D

// Generate the mesh
Mesh 2;
Save "square9.msh";
