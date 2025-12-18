// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <dolfin/fem/Form.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "VD_NSEPressEquation.h"

#include "ufl/VD_Pressure_2D_P1P1.h"
#include "ufl/VD_Pressure_2D_P2P1.h"
#include "ufl/VD_Pressure_2D_P3P2.h"
#include "ufl/VD_Pressure_2D_P3P3.h"
#include "ufl/VD_Pressure_2D_P4P3.h"
#include "ufl/VD_Pressure_3D_P1P1.h"
#include "ufl/VD_Pressure_3D_P2P1.h"
#include "ufl/VD_Pressure_3D_P3P2.h"
#include "ufl/VD_Pressure_3D_P3P3.h"
#include "ufl/VD_Pressure_3D_P4P3.h"

using namespace fem_varden_flow;
using namespace dolfin;

namespace {

#define DEFINE_PRESSURE_TRAITS(NAME)                                            \
struct NAME##_Traits {                                                          \
  using FunctionSpace = NAME::Form_a_outflow_FunctionSpace_0;                   \
  using Form_a_outflow = NAME::Form_a_outflow;                                  \
  using Form_L_outflow = NAME::Form_L_outflow;                                  \
  using Form_ai = NAME::Form_ai;                                                \
  using Form_Li = NAME::Form_Li;                                                \
};

DEFINE_PRESSURE_TRAITS(VD_Pressure_2D_P1P1)
DEFINE_PRESSURE_TRAITS(VD_Pressure_2D_P2P1)
DEFINE_PRESSURE_TRAITS(VD_Pressure_2D_P3P2)
DEFINE_PRESSURE_TRAITS(VD_Pressure_2D_P3P3)
DEFINE_PRESSURE_TRAITS(VD_Pressure_2D_P4P3)
DEFINE_PRESSURE_TRAITS(VD_Pressure_3D_P1P1)
DEFINE_PRESSURE_TRAITS(VD_Pressure_3D_P2P1)
DEFINE_PRESSURE_TRAITS(VD_Pressure_3D_P3P2)
DEFINE_PRESSURE_TRAITS(VD_Pressure_3D_P3P3)
DEFINE_PRESSURE_TRAITS(VD_Pressure_3D_P4P3)

#undef DEFINE_PRESSURE_TRAITS

template <typename Traits, typename... Args>
std::shared_ptr<FunctionSpace>
make_pressure_space(const std::shared_ptr<const Mesh> &mesh,
                    Args &&...args) {
  return std::make_shared<typename Traits::FunctionSpace>(
      mesh, std::forward<Args>(args)...);
}

template <typename Traits>
void configure_pressure_system(System &eqi, System &eq_outflow) {
  eq_outflow.a = std::make_shared<typename Traits::Form_a_outflow>(eq_outflow.Q,
                                                                   eq_outflow.Q);
  eq_outflow.L =
      std::make_shared<typename Traits::Form_L_outflow>(eq_outflow.Q);
  eqi.a = std::make_shared<typename Traits::Form_ai>(eq_outflow.Q,
                                                     eq_outflow.Q);
  eqi.L = std::make_shared<typename Traits::Form_Li>(eq_outflow.Q);
}

[[noreturn]] void throw_unsupported(std::size_t dim, int vm_degree,
                                    int q_degree) {
  throw std::invalid_argument("Unsupported pressure setup: dim=" +
                              std::to_string(dim) +
                              ", PolDegree_V_M=" + std::to_string(vm_degree) +
                              ", PolDegree_Q=" + std::to_string(q_degree));
}

} // namespace

//-----------------------------------------------------------------------------
void VD_NSEPressEquation::init(std::shared_ptr<const Mesh> mesh, System &eqi,
                               System &eq_outflow, Parameters &p) {
  const std::size_t D = mesh->topology().dim();
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];
  ///

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        configure_pressure_system<VD_Pressure_2D_P1P1_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_pressure_system<VD_Pressure_2D_P2P1_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_pressure_system<VD_Pressure_2D_P3P2_Traits>(eqi, eq_outflow);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_pressure_system<VD_Pressure_2D_P3P3_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_pressure_system<VD_Pressure_2D_P4P3_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  if (D == 3) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        configure_pressure_system<VD_Pressure_3D_P1P1_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_pressure_system<VD_Pressure_3D_P2P1_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_pressure_system<VD_Pressure_3D_P3P2_Traits>(eqi, eq_outflow);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_pressure_system<VD_Pressure_3D_P3P3_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_pressure_system<VD_Pressure_3D_P4P3_Traits>(eqi, eq_outflow);
        return;
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  throw std::invalid_argument("Unsupported mesh dimension: " +
                              std::to_string(D));
}

//-----------------------------------------------------------------------------
void VD_NSEPressEquation::init(std::shared_ptr<const Mesh> mesh,
                               std::shared_ptr<const FunctionSpace> Q,
                               System &eqi, System &eq_outflow,
                               Parameters &p) {
  const std::size_t D = mesh->topology().dim();

  eqi.Q = Q;
  eq_outflow.Q = Q;
  VD_NSEPressEquation::init(mesh, eqi, eq_outflow, p);
}

std::shared_ptr<FunctionSpace>
VD_NSEPressEquation::create_pressure_space(std::shared_ptr<const Mesh> mesh,
                                           Parameters &p) {
  const std::size_t D = mesh->topology().dim();
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_2D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_2D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_pressure_space<VD_Pressure_2D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_2D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_2D_P4P3_Traits>(mesh);
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  if (D == 3) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_3D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_3D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_pressure_space<VD_Pressure_3D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_3D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_3D_P4P3_Traits>(mesh);
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  throw std::invalid_argument("Unsupported mesh dimension: " +
                              std::to_string(D));
}

std::shared_ptr<FunctionSpace> VD_NSEPressEquation::create_pressure_space(
    std::shared_ptr<const Mesh> mesh,
    std::shared_ptr<const SubDomain> periodic,
    Parameters &p) {
  if (!periodic) {
    throw std::invalid_argument(
        "Periodic boundary subdomain must not be null when creating periodic spaces");
  }

  const std::size_t D = mesh->topology().dim();
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_2D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_2D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_pressure_space<VD_Pressure_2D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_2D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_2D_P4P3_Traits>(mesh, periodic);
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  if (D == 3) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_3D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_pressure_space<VD_Pressure_3D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_pressure_space<VD_Pressure_3D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_3D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_pressure_space<VD_Pressure_3D_P4P3_Traits>(mesh, periodic);
      }
      break;
    default:
      break;
    }
    throw_unsupported(D, pol_degree_V_M, pol_degree_Q);
  }

  throw std::invalid_argument("Unsupported mesh dimension: " +
                              std::to_string(D));
}
