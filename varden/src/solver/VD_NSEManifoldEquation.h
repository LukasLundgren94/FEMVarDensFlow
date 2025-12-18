// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_MANIFOLD_EQUATION_H
#define __VD_NSE_MANIFOLD_EQUATION_H

#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/SubDomain.h>

#include "System.h"

///
namespace dolfin {
class Mesh;
}

////
using dolfin::Function;
using dolfin::Mesh;
using dolfin::BoundaryMesh;
using dolfin::Parameters;
using dolfin::Variable;

namespace fem_varden_flow {
class VD_NSEManifoldEquation : public Variable {
public:
  ///
  static void init(std::shared_ptr<const BoundaryMesh> mesh, System &eq, Parameters &p);

  ///
  static void init(std::shared_ptr<const BoundaryMesh> mesh,
                   std::shared_ptr<const FunctionSpace> M, System &eq,
                   Parameters &p);
};
} // namespace fem_varden_flow
#endif
