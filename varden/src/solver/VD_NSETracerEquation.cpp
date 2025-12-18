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

#include "VD_NSETracerEquation.h"

#include "ufl/VD_Tracer_2D_P1P1.h"
#include "ufl/VD_Tracer_2D_P2P1.h"
#include "ufl/VD_Tracer_2D_P3P2.h"
#include "ufl/VD_Tracer_2D_P3P3.h"
#include "ufl/VD_Tracer_2D_P4P3.h"
#include "ufl/VD_Tracer_3D_P1P1.h"
#include "ufl/VD_Tracer_3D_P2P1.h"
#include "ufl/VD_Tracer_3D_P3P2.h"
#include "ufl/VD_Tracer_3D_P3P3.h"
#include "ufl/VD_Tracer_3D_P4P3.h"

#include "System.h"

using namespace fem_varden_flow;
using namespace dolfin;

namespace {

#define DEFINE_TRACER_TRAITS(NAME)                                              \
struct NAME##_Traits {                                                          \
  using FunctionSpace = NAME::FunctionSpace;                                    \
  using Form_a = NAME::Form_a;                                                  \
  using Form_L = NAME::Form_L;                                                  \
};

DEFINE_TRACER_TRAITS(VD_Tracer_2D_P1P1)
DEFINE_TRACER_TRAITS(VD_Tracer_2D_P2P1)
DEFINE_TRACER_TRAITS(VD_Tracer_2D_P3P2)
DEFINE_TRACER_TRAITS(VD_Tracer_2D_P3P3)
DEFINE_TRACER_TRAITS(VD_Tracer_2D_P4P3)
DEFINE_TRACER_TRAITS(VD_Tracer_3D_P1P1)
DEFINE_TRACER_TRAITS(VD_Tracer_3D_P2P1)
DEFINE_TRACER_TRAITS(VD_Tracer_3D_P3P2)
DEFINE_TRACER_TRAITS(VD_Tracer_3D_P3P3)
DEFINE_TRACER_TRAITS(VD_Tracer_3D_P4P3)

#undef DEFINE_TRACER_TRAITS

template <typename Traits, typename... Args>
std::shared_ptr<FunctionSpace>
make_tracer_space(const std::shared_ptr<const Mesh> &mesh, Args &&...args) {
  return std::make_shared<typename Traits::FunctionSpace>(
      mesh, std::forward<Args>(args)...);
}

template <typename Traits, typename... Args>
void configure_tracer_system(System &eq,
                             const std::shared_ptr<const Mesh> &mesh,
                             Args &&...args) {
  if (!eq.M) {
    eq.M = make_tracer_space<Traits>(mesh, std::forward<Args>(args)...);
  }
  eq.a = std::make_shared<typename Traits::Form_a>(eq.M, eq.M);
  eq.L = std::make_shared<typename Traits::Form_L>(eq.M);
}

[[noreturn]] void throw_unsupported(std::size_t dim, int vm_degree,
                                    int q_degree) {
  throw std::invalid_argument("Unsupported tracer setup: dim=" +
                              std::to_string(dim) + ", PolDegree_V_M=" +
                              std::to_string(vm_degree) +
                              ", PolDegree_Q=" + std::to_string(q_degree));
}

} // namespace

//-----------------------------------------------------------------------------
void VD_NSETracerEquation::init(std::shared_ptr<const Mesh> mesh,
                                   System &eq, Parameters &p) {
  const std::size_t D = mesh->topology().dim();
  const int pol_degree_V_M = p["PolDegree_V_M"];
  const int pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        configure_tracer_system<VD_Tracer_2D_P1P1_Traits>(eq, mesh);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_tracer_system<VD_Tracer_2D_P2P1_Traits>(eq, mesh);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_tracer_system<VD_Tracer_2D_P3P2_Traits>(eq, mesh);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_tracer_system<VD_Tracer_2D_P3P3_Traits>(eq, mesh);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_tracer_system<VD_Tracer_2D_P4P3_Traits>(eq, mesh);
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
        configure_tracer_system<VD_Tracer_3D_P1P1_Traits>(eq, mesh);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_tracer_system<VD_Tracer_3D_P2P1_Traits>(eq, mesh);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_tracer_system<VD_Tracer_3D_P3P2_Traits>(eq, mesh);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_tracer_system<VD_Tracer_3D_P3P3_Traits>(eq, mesh);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_tracer_system<VD_Tracer_3D_P4P3_Traits>(eq, mesh);
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
void VD_NSETracerEquation::init(std::shared_ptr<const Mesh> mesh,
                                   std::shared_ptr<const FunctionSpace> M,
                                   System &eq, Parameters &p) {
  const std::size_t D = mesh->topology().dim();

  eq.M = M;
  VD_NSETracerEquation::init(mesh, eq, p);
}

std::shared_ptr<FunctionSpace>
VD_NSETracerEquation::create_tracer_space(std::shared_ptr<const Mesh> mesh,
                                          Parameters &p) {
  const std::size_t D = mesh->topology().dim();
  const int pol_degree_V_M = p["PolDegree_V_M"];
  const int pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_2D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_2D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_tracer_space<VD_Tracer_2D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_2D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_2D_P4P3_Traits>(mesh);
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
        return make_tracer_space<VD_Tracer_3D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_3D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_tracer_space<VD_Tracer_3D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_3D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_3D_P4P3_Traits>(mesh);
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

std::shared_ptr<FunctionSpace> VD_NSETracerEquation::create_tracer_space(
    std::shared_ptr<const Mesh> mesh,
    std::shared_ptr<const SubDomain> periodic,
    Parameters &p) {
  if (!periodic) {
    throw std::invalid_argument(
        "Periodic boundary subdomain must not be null when creating periodic spaces");
  }

  const std::size_t D = mesh->topology().dim();
  const int pol_degree_V_M = p["PolDegree_V_M"];
  const int pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_2D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_2D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_tracer_space<VD_Tracer_2D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_2D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_2D_P4P3_Traits>(mesh, periodic);
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
        return make_tracer_space<VD_Tracer_3D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_tracer_space<VD_Tracer_3D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_tracer_space<VD_Tracer_3D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_3D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_tracer_space<VD_Tracer_3D_P4P3_Traits>(mesh, periodic);
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
