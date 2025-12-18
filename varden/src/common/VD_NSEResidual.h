// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_RESIDUAL_H
#define __VD_NSE_RESIDUAL_H

#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/SubDomain.h>
#include <stdexcept>
#include <string>

#include "System.h"

#include "ufl/VD_Residual_2D_P1P1.h"
#include "ufl/VD_Residual_2D_P2P1.h"
#include "ufl/VD_Residual_2D_P2P2.h"
#include "ufl/VD_Residual_2D_P3P2.h"
#include "ufl/VD_Residual_2D_P3P3.h"
#include "ufl/VD_Residual_2D_P4P3.h"

#include "ufl/VD_Residual_3D_P1P1.h"
#include "ufl/VD_Residual_3D_P2P1.h"
#include "ufl/VD_Residual_3D_P2P2.h"
#include "ufl/VD_Residual_3D_P3P2.h"
#include "ufl/VD_Residual_3D_P3P3.h"
#include "ufl/VD_Residual_3D_P4P3.h"


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

namespace detail {

#define DEFINE_RESIDUAL_TRAITS(NAME)                                            \
struct NAME##_Traits {                                                          \
  using FunctionSpace = NAME::FunctionSpace;                                    \
  using LinearForm = NAME::LinearForm;                                          \
  using BilinearForm = NAME::BilinearForm;                                      \
};

DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P1P1)
DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P2P1)
DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P2P2)
DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P3P2)
DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P3P3)
DEFINE_RESIDUAL_TRAITS(VD_Residual_2D_P4P3)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P1P1)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P2P1)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P2P2)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P3P2)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P3P3)
DEFINE_RESIDUAL_TRAITS(VD_Residual_3D_P4P3)

#undef DEFINE_RESIDUAL_TRAITS

template <typename Traits>
inline void configure_residual_system(System &eq,
                                      const std::shared_ptr<const Mesh> &mesh) {
  if (!eq.M) {
    eq.M = std::make_shared<typename Traits::FunctionSpace>(mesh);
  }
  eq.L = std::make_shared<typename Traits::LinearForm>(eq.M);
  eq.a = std::make_shared<typename Traits::BilinearForm>(eq.M, eq.M);
}

[[noreturn]] inline void throw_unsupported(std::size_t dim, int vm_degree,
                                           int q_degree) {
  throw std::invalid_argument("Unsupported residual setup: dim=" +
                              std::to_string(dim) +
                              ", PolDegree_V_M=" + std::to_string(vm_degree) +
                              ", PolDegree_Q=" + std::to_string(q_degree));
}

} // namespace detail


class VD_NSEResidual : public Variable {
public:

  static void init(std::shared_ptr<const Mesh> mesh, System &eq,
                        Parameters &p) {
    const std::size_t D = mesh->topology().dim();

    int const pol_degree_V_M = p["PolDegree_V_M"];
    int const pol_degree_Q = p["PolDegree_Q"];
    ///
    if (D == 2) {
      switch (pol_degree_V_M) {
      case 1:
        if (pol_degree_Q == 1) {
          detail::configure_residual_system<detail::VD_Residual_2D_P1P1_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 2:
        if (pol_degree_Q == 1) {
          detail::configure_residual_system<detail::VD_Residual_2D_P2P1_Traits>(eq,
                                                                       mesh);
          return;
        }
        if (pol_degree_Q == 2) {
          detail::configure_residual_system<detail::VD_Residual_2D_P2P2_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 3:
        if (pol_degree_Q == 2) {
          detail::configure_residual_system<detail::VD_Residual_2D_P3P2_Traits>(eq,
                                                                       mesh);
          return;
        }
        if (pol_degree_Q == 3) {
          detail::configure_residual_system<detail::VD_Residual_2D_P3P3_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 4:
        if (pol_degree_Q == 3) {
          detail::configure_residual_system<detail::VD_Residual_2D_P4P3_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      default:
        break;
      }
      detail::throw_unsupported(D, pol_degree_V_M, pol_degree_Q);

    } else if (D == 3) {
      switch (pol_degree_V_M) {
      case 1:
        if (pol_degree_Q == 1) {
          detail::configure_residual_system<detail::VD_Residual_3D_P1P1_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 2:
        if (pol_degree_Q == 1) {
          detail::configure_residual_system<detail::VD_Residual_3D_P2P1_Traits>(eq,
                                                                       mesh);
          return;
        }
        if (pol_degree_Q == 2) {
          detail::configure_residual_system<detail::VD_Residual_3D_P2P2_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 3:
        if (pol_degree_Q == 2) {
          detail::configure_residual_system<detail::VD_Residual_3D_P3P2_Traits>(eq,
                                                                       mesh);
          return;
        }
        if (pol_degree_Q == 3) {
          detail::configure_residual_system<detail::VD_Residual_3D_P3P3_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      case 4:
        if (pol_degree_Q == 3) {
          detail::configure_residual_system<detail::VD_Residual_3D_P4P3_Traits>(eq,
                                                                       mesh);
          return;
        }
        break;
      default:
        break;
      }
      detail::throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
    } else {
      detail::throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
    }
    // assemble(eq.A, *eq.a);
  }

  //-----------------------------------------------------------------------------
  static void init(std::shared_ptr<const Mesh> mesh,
                   std::shared_ptr<const FunctionSpace> M, System &eq,
                   Parameters &p) {
    eq.M = M;
    init(mesh, eq, p);
  }

};
} // namespace fem_varden_flow
#endif
