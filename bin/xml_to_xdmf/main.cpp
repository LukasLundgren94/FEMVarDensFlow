// Copyright (C) 2014 Murtazo Nazarov
//
//
//
//

#include <dolfin.h>

using namespace dolfin;

int main(int argc, char** argv)
{
  std::cout << "Converting:  " << argv[1] 
	    << "---->" << argv[2] << std::endl;
  
  auto mesh = std::make_shared<Mesh>(argv[1]);

  XDMFFile xmesh(mesh->mpi_comm(), argv[2]);
  xmesh.write(*mesh);

  return 0;

}
