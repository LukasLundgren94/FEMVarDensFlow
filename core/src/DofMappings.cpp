// Copyright (C) 2015 Murtazo Nazarov
// Licensed under the MIT License – see the LICENSE file for details

#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/DofMap.h>

#include "DofMappings.h"

using namespace fem_varden_flow;

//-------------------------------------------------------------------------------
DofMappings::DofMappings()
{
}

//-------------------------------------------------------------------------------
DofMappings::~DofMappings()
{
}

//-------------------------------------------------------------------------------
void DofMappings::get_all_dofs(Function const& w,
                               std::vector<dolfin::la_index>& dof)
{
  std::size_t w_max_el_dofs = w.function_space()->dofmap()->max_element_dofs();

  Mesh const& mesh(*w.function_space()->mesh());
  dof.resize(mesh.num_cells() * w_max_el_dofs);

  // needed to update ufc::cell
  std::vector<double> coordinate_dofs;

  std::size_t ii = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    cell->get_vertex_coordinates(coordinate_dofs);

    auto cell_dofs_w =
        w.function_space()->dofmap()->cell_dofs((*cell).index());
    for (long int i = 0; i < cell_dofs_w.size(); ++i, ++ii)
    {
      dof[ii] = cell_dofs_w[i];
    }
  }
}
