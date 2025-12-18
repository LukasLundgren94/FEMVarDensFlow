// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_MOM_EQUATION_H
#define __VD_NSE_MOM_EQUATION_H

#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/SubDomain.h>
#include <memory>

#include "System.h"

///
namespace dolfin
{
  class Mesh;
  class FunctionSpace;
  class Form;
}

using dolfin::Mesh;
using dolfin::Form;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::Variable;
using dolfin::Parameters;
using dolfin::SubDomain;
using dolfin::Array;

namespace fem_varden_flow
{
  class VD_NSEMomEquation
  {
  public:

  static void init_mixed(
    std::shared_ptr<const Mesh> mesh, std::shared_ptr<const FunctionSpace> V,
    std::shared_ptr<const FunctionSpace> Q, System &eq, Parameters &p, std::shared_ptr<Form> &a01, std::shared_ptr<Form> &a10, std::shared_ptr<Form> &a11, std::shared_ptr<Form> &as, System &eq_vloc, System &eq_lps, System &eq_lps_tensor, System &eq_h);

  static void init_mixed(std::shared_ptr<const Mesh> mesh, System &eq,
                       Parameters &p, std::shared_ptr<Form> &a01, std::shared_ptr<Form> &a10, std::shared_ptr<Form> &a11, std::shared_ptr<Form> &as, System &eq_vloc, System &eq_lps, System &eq_lps_tensor, System &eq_h);

  static std::shared_ptr<FunctionSpace>
  create_velocity_space(std::shared_ptr<const Mesh> mesh, Parameters &p);

  static std::shared_ptr<FunctionSpace> create_velocity_space(
      std::shared_ptr<const Mesh> mesh,
      std::shared_ptr<const SubDomain> periodic,
      Parameters &p);

};
}
#endif
