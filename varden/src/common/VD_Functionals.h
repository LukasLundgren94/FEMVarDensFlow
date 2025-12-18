// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_FUNCTIONALS_H
#define __VD_NSE_FUNCTIONALS_H

#include <dolfin/fem/Form.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/SubDomain.h>
#include <stdexcept>
#include <string>

#include "System.h"
#include "ufl/VD_Functionals_2D_P1P1.h"
#include "ufl/VD_Functionals_2D_P2P1.h"
#include "ufl/VD_Functionals_2D_P2P2.h"
#include "ufl/VD_Functionals_2D_P3P2.h"
#include "ufl/VD_Functionals_2D_P3P3.h"
#include "ufl/VD_Functionals_2D_P4P3.h"

#include "ufl/VD_Functionals_3D_P1P1.h"
#include "ufl/VD_Functionals_3D_P2P1.h"
#include "ufl/VD_Functionals_3D_P2P2.h"
#include "ufl/VD_Functionals_3D_P3P2.h"
#include "ufl/VD_Functionals_3D_P3P3.h"
#include "ufl/VD_Functionals_3D_P4P3.h"

///
namespace dolfin {
class Mesh;
}

///
using dolfin::Function;
using dolfin::Mesh;
using dolfin::Parameters;
using dolfin::Variable;

using namespace dolfin;

namespace fem_varden_flow {

class VD_Functionals;

namespace detail {

#define DEFINE_FUNCTIONAL_TRAITS(NAME)                                          \
struct NAME##_Traits {                                                          \
  using Form_Pavg = NAME::Form_Pavg;                                            \
  using Form_momentum_ = NAME::Form_momentum_;                                  \
  using Form_kinetic_energy = NAME::Form_kinetic_energy;                        \
  using Form_L2rho = NAME::Form_L2rho;                                          \
  using Form_mass = NAME::Form_mass;                                            \
  using Form_angular_momentum_z = NAME::Form_angular_momentum_z;                \
  using Form_offset = NAME::Form_offset;                                        \
  using Form_melt_rate = NAME::Form_melt_rate;                                  \
  using Form_tau0 = NAME::Form_tau0;                                            \
};

DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P1P1)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P2P1)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P2P2)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P3P2)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P3P3)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_2D_P4P3)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P1P1)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P2P1)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P2P2)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P3P2)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P3P3)
DEFINE_FUNCTIONAL_TRAITS(VD_Functionals_3D_P4P3)

#undef DEFINE_FUNCTIONAL_TRAITS

template <typename Traits>
void configure_functionals(VD_Functionals &fn,
                           const std::shared_ptr<const Mesh> &mesh);

[[noreturn]] inline void throw_functionals_unsupported(std::size_t dim, int vm_degree,
                                           int q_degree) {
  throw std::invalid_argument("Unsupported functional setup: dim=" +
                              std::to_string(dim) +
                              ", PolDegree_V_M=" + std::to_string(vm_degree) +
                              ", PolDegree_Q=" + std::to_string(q_degree));
}

} // namespace detail

class VD_Functionals : public Variable {
public:
  std::shared_ptr<Form> Pavg;
  std::shared_ptr<Form> momentum_;
  std::shared_ptr<Form> kinetic_energy;
  std::shared_ptr<Form> L2rho;
  std::shared_ptr<Form> mass;
  std::shared_ptr<Form> angular_momentum_z;
  std::shared_ptr<Form> offset;
  std::shared_ptr<Form> melt_rate;
  std::shared_ptr<Form> tau0;

  VD_Functionals(std::shared_ptr<const Mesh> mesh,
                                 Parameters &p) {
    init(mesh, p);
  }

  ///
  void init(std::shared_ptr<const Mesh> mesh, Parameters &p) {
    const std::size_t D = mesh->topology().dim();

    int const pol_degree_V_M = p["PolDegree_V_M"];
    int const pol_degree_Q = p["PolDegree_Q"];
    ///
    if (D == 2) {

      switch (pol_degree_V_M) {
      case 1:
        if (pol_degree_Q == 1) {
          detail::configure_functionals<detail::VD_Functionals_2D_P1P1_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 2:
        if (pol_degree_Q == 1) {
          detail::configure_functionals<detail::VD_Functionals_2D_P2P1_Traits>(*this,
                                                                      mesh);
          return;
        }
        if (pol_degree_Q == 2) {
          detail::configure_functionals<detail::VD_Functionals_2D_P2P2_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 3:
        if (pol_degree_Q == 2) {
          detail::configure_functionals<detail::VD_Functionals_2D_P3P2_Traits>(*this,
                                                                      mesh);
          return;
        }
        if (pol_degree_Q == 3) {
          detail::configure_functionals<detail::VD_Functionals_2D_P3P3_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 4:
        if (pol_degree_Q == 3) {
          detail::configure_functionals<detail::VD_Functionals_2D_P4P3_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      default:
        break;
      }
      detail::throw_functionals_unsupported(D, pol_degree_V_M, pol_degree_Q);

    } else if (D == 3) {
      switch (pol_degree_V_M) {
      case 1:
        if (pol_degree_Q == 1) {
          detail::configure_functionals<detail::VD_Functionals_3D_P1P1_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 2:
        if (pol_degree_Q == 1) {
          detail::configure_functionals<detail::VD_Functionals_3D_P2P1_Traits>(*this,
                                                                      mesh);
          return;
        }
        if (pol_degree_Q == 2) {
          detail::configure_functionals<detail::VD_Functionals_3D_P2P2_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 3:
        if (pol_degree_Q == 2) {
          detail::configure_functionals<detail::VD_Functionals_3D_P3P2_Traits>(*this,
                                                                      mesh);
          return;
        }
        if (pol_degree_Q == 3) {
          detail::configure_functionals<detail::VD_Functionals_3D_P3P3_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      case 4:
        if (pol_degree_Q == 3) {
          detail::configure_functionals<detail::VD_Functionals_3D_P4P3_Traits>(*this,
                                                                      mesh);
          return;
        }
        break;
      default:
        break;
      }
      detail::throw_functionals_unsupported(D, pol_degree_V_M, pol_degree_Q);
    } else {
      detail::throw_functionals_unsupported(D, pol_degree_V_M, pol_degree_Q);
    }
  }
};

namespace detail {

template <typename Traits>
inline void configure_functionals(VD_Functionals &fn,
                                  const std::shared_ptr<const Mesh> &mesh) {
  fn.Pavg = std::make_shared<typename Traits::Form_Pavg>(mesh);
  fn.momentum_ = std::make_shared<typename Traits::Form_momentum_>(mesh);
  fn.kinetic_energy =
      std::make_shared<typename Traits::Form_kinetic_energy>(mesh);
  fn.L2rho = std::make_shared<typename Traits::Form_L2rho>(mesh);
  fn.mass = std::make_shared<typename Traits::Form_mass>(mesh);
  fn.angular_momentum_z =
      std::make_shared<typename Traits::Form_angular_momentum_z>(mesh);
  fn.offset = std::make_shared<typename Traits::Form_offset>(mesh);
  fn.melt_rate = std::make_shared<typename Traits::Form_melt_rate>(mesh);
  fn.tau0 = std::make_shared<typename Traits::Form_tau0>(mesh);
}

} // namespace detail
} // namespace fem_varden_flow
#endif
