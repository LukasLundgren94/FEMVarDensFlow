// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_PRESS_EQUATION_H
#define __VD_NSE_PRESS_EQUATION_H

#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/SubDomain.h>
#include <string>

#include "System.h"

///
namespace dolfin {
class Mesh;
class FunctionSpace;
class SubDomain;
}

///
using dolfin::Function;
using dolfin::Mesh;
using dolfin::FunctionSpace;
using dolfin::Parameters;
using dolfin::Variable;
using dolfin::SubDomain;

namespace fem_varden_flow {
class VD_NSEPressEquation : public Variable {
public:
  ///
  static void init(std::shared_ptr<const Mesh> mesh, System &eqi,
                   System &eq_outflow, 
                   Parameters &p);

  ///
  static void init(std::shared_ptr<const Mesh> mesh,
                   std::shared_ptr<const FunctionSpace> Q, System &eqi,
                   System &eq_outflow, 
                    Parameters &p);

  static std::shared_ptr<FunctionSpace>
  create_pressure_space(std::shared_ptr<const Mesh> mesh, Parameters &p);

  static std::shared_ptr<FunctionSpace> create_pressure_space(
      std::shared_ptr<const Mesh> mesh,
      std::shared_ptr<const SubDomain> periodic,
      Parameters &p);

  ///
};
} // namespace fem_varden_flow
#endif
