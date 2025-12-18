// Copyright (C) 2015 Murtazo Nazarov
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __DOF_MAPPINGS_H
#define __DOF_MAPPINGS_H

#include <dolfin/function/Function.h>
#include <dolfin/mesh/Mesh.h>


///
namespace dolfin
{
  class Mesh;
  class Function;
  class Cell;
  class FunctionSpace;
  class DofMap;
}

using dolfin::Mesh;
using dolfin::Function;
using dolfin::Cell;
using dolfin::FunctionSpace;
using dolfin::DofMap;
using dolfin::CellIterator;

///
namespace fem_varden_flow
{
  class DofMappings
  {
  public:

    ///
    DofMappings();

    ///
    ~DofMappings();

    ///
    static void get_all_dofs(Function const& w, std::vector<dolfin::la_index>& dof);
    
  };
}

#endif
