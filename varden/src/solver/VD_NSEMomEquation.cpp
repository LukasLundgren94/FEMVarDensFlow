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

#include "VD_NSEMomEquation.h"

#include "ufl/VD_Mixed_2D_P1P1.h"
#include "ufl/VD_Mixed_2D_P2P1.h"
#include "ufl/VD_Mixed_2D_P3P2.h"
#include "ufl/VD_Mixed_2D_P3P3.h"
#include "ufl/VD_Mixed_2D_P4P3.h"
#include "ufl/VD_Mixed_3D_P1P1.h"
#include "ufl/VD_Mixed_3D_P2P1.h"
#include "ufl/VD_Mixed_3D_P3P2.h"
#include "ufl/VD_Mixed_3D_P3P3.h"
#include "ufl/VD_Mixed_3D_P4P3.h"

using namespace fem_varden_flow;
using namespace dolfin;

namespace {

#define DEFINE_MIXED_TRAITS(NAME)                                               \
struct NAME##_Traits {                                                          \
  using CoefficientSpace_u_n = NAME::CoefficientSpace_u_n;                      \
  using Form_a11_FunctionSpace_0 = NAME::Form_a11_FunctionSpace_0;                      \
  using Form_a_LPS_FunctionSpace_0 = NAME::Form_a_LPS_FunctionSpace_0;          \
  using Form_a_LPS_tensor_FunctionSpace_0 =                                       \
      NAME::Form_a_LPS_tensor_FunctionSpace_0;                                   \
  using CoefficientSpace_rho_n = NAME::CoefficientSpace_rho_n;                  \
  using Form_a00 = NAME::Form_a00;                                              \
  using Form_b0 = NAME::Form_b0;                                                \
  using Form_a01 = NAME::Form_a01;                                              \
  using Form_a10 = NAME::Form_a10;                                              \
  using Form_a11 = NAME::Form_a11;                                              \
  using CoefficientSpace_rho_K = NAME::CoefficientSpace_rho_K;                  \
  using CoefficientSpace_h_grad_phi = NAME::CoefficientSpace_h_grad_phi;        \
  using Form_aSchur = NAME::Form_aSchur;                                        \
  using Form_a_proj_U = NAME::Form_a_proj_U;                                    \
  using Form_L_proj_U = NAME::Form_L_proj_U;                                    \
  using Form_a_LPS = NAME::Form_a_LPS;                                          \
  using Form_L_LPS = NAME::Form_L_LPS;                                          \
  using Form_a_LPS_tensor = NAME::Form_a_LPS_tensor;                            \
  using Form_L_LPS_tensor = NAME::Form_L_LPS_tensor;                            \
  using Form_a_h = NAME::Form_a_h;                                              \
  using Form_L_h = NAME::Form_L_h;                                              \
};

DEFINE_MIXED_TRAITS(VD_Mixed_2D_P1P1)
DEFINE_MIXED_TRAITS(VD_Mixed_2D_P2P1)
DEFINE_MIXED_TRAITS(VD_Mixed_2D_P3P2)
DEFINE_MIXED_TRAITS(VD_Mixed_2D_P3P3)
DEFINE_MIXED_TRAITS(VD_Mixed_2D_P4P3)
DEFINE_MIXED_TRAITS(VD_Mixed_3D_P1P1)
DEFINE_MIXED_TRAITS(VD_Mixed_3D_P2P1)
DEFINE_MIXED_TRAITS(VD_Mixed_3D_P3P2)
DEFINE_MIXED_TRAITS(VD_Mixed_3D_P3P3)
DEFINE_MIXED_TRAITS(VD_Mixed_3D_P4P3)

#undef DEFINE_MIXED_TRAITS

template <typename Mixed, typename... Args>
std::shared_ptr<FunctionSpace>
make_velocity_space(const std::shared_ptr<const Mesh> &mesh,
                    Args &&...args) {
  return std::make_shared<typename Mixed::CoefficientSpace_u_n>(
      mesh, std::forward<Args>(args)...);
}

template <typename Mixed>
void configure_momentum_system(System &eq,
                               const std::shared_ptr<const Mesh> &mesh,
                               std::shared_ptr<Form> &a01,
                               std::shared_ptr<Form> &a10,
                               std::shared_ptr<Form> &a11,
                               std::shared_ptr<Form> &as, System &eq_vloc,
                               System &eq_lps, System &eq_lps_tensor,
                               System &eq_h) {
  if (!eq.V) {
    eq.V = std::make_shared<typename Mixed::CoefficientSpace_u_n>(mesh);
  }
  if (!eq.Q) {
    eq.Q = std::make_shared<typename Mixed::Form_a11_FunctionSpace_0>(mesh);
  }
  if (!eq_lps.V) {
    eq_lps.V = std::make_shared<typename Mixed::Form_a_LPS_FunctionSpace_0>(
        mesh);
  }
  if (!eq_lps_tensor.V) {
    eq_lps_tensor.V = std::make_shared<
        typename Mixed::Form_a_LPS_tensor_FunctionSpace_0>(mesh);
  }
  if (!eq.M) {
    eq.M = std::make_shared<typename Mixed::CoefficientSpace_rho_n>(mesh);
  }

  eq.a = std::make_shared<typename Mixed::Form_a00>(eq.V, eq.V);
  eq.L = std::make_shared<typename Mixed::Form_b0>(eq.V);
  a01 = std::make_shared<typename Mixed::Form_a01>(eq.Q, eq.V);
  a10 = std::make_shared<typename Mixed::Form_a10>(eq.V, eq.Q);
  a11 = std::make_shared<typename Mixed::Form_a11>(eq.Q, eq.Q);
  eq.DG0 = std::make_shared<typename Mixed::CoefficientSpace_rho_K>(mesh);
  eq.V_DG0 =
      std::make_shared<typename Mixed::CoefficientSpace_h_grad_phi>(mesh);
  as = std::make_shared<typename Mixed::Form_aSchur>(eq.Q, eq.Q);

  eq_vloc.a =
      std::make_shared<typename Mixed::Form_a_proj_U>(eq.M, eq.M);
  eq_vloc.L = std::make_shared<typename Mixed::Form_L_proj_U>(eq.M);

  eq_lps.a = std::make_shared<typename Mixed::Form_a_LPS>(eq_lps.V, eq_lps.V);
  eq_lps.L = std::make_shared<typename Mixed::Form_L_LPS>(eq_lps.V);

  eq_lps_tensor.a = std::make_shared<typename Mixed::Form_a_LPS_tensor>(
      eq_lps_tensor.V, eq_lps_tensor.V);
  eq_lps_tensor.L = std::make_shared<typename Mixed::Form_L_LPS_tensor>(
      eq_lps_tensor.V);

  eq_h.a = std::make_shared<typename Mixed::Form_a_h>(eq.V, eq.V);
  eq_h.L = std::make_shared<typename Mixed::Form_L_h>(eq.V);
}

[[noreturn]] void throw_unsupported(std::size_t dim, int vm_degree,
                                    int q_degree) {
  throw std::invalid_argument("Unsupported momentum setup: dim=" +
                              std::to_string(dim) +
                              ", PolDegree_V_M=" + std::to_string(vm_degree) +
                              ", PolDegree_Q=" + std::to_string(q_degree));
}

} // namespace

void VD_NSEMomEquation::init_mixed(
    std::shared_ptr<const Mesh> mesh, std::shared_ptr<const FunctionSpace> V,
    std::shared_ptr<const FunctionSpace> Q, System &eq, Parameters &p,
    std::shared_ptr<Form> &a01, std::shared_ptr<Form> &a10,
    std::shared_ptr<Form> &a11, std::shared_ptr<Form> &as, System &eq_vloc,
    System &eq_lps, System &eq_lps_tensor, System &eq_h) {
  eq.V = V;
  eq.Q = Q;
  eq.M = Q;

  VD_NSEMomEquation::init_mixed(mesh, eq, p, a01, a10, a11, as, eq_vloc, eq_lps,
                                eq_lps_tensor, eq_h);
  return;
}

void VD_NSEMomEquation::init_mixed(std::shared_ptr<const Mesh> mesh, System &eq,
                                   Parameters &p, std::shared_ptr<Form> &a01,
                                   std::shared_ptr<Form> &a10,
                                   std::shared_ptr<Form> &a11,
                                   std::shared_ptr<Form> &as, System &eq_vloc,
                                   System &eq_lps, System &eq_lps_tensor,
                                   System &eq_h) {
  const std::size_t D = mesh->topology().dim();
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];
  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        configure_momentum_system<VD_Mixed_2D_P1P1_Traits>(eq, mesh, a01, a10, a11, as,
                                                    eq_vloc, eq_lps,
                                                    eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_momentum_system<VD_Mixed_2D_P2P1_Traits>(eq, mesh, a01, a10, a11, as,
                                                    eq_vloc, eq_lps,
                                                    eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_momentum_system<VD_Mixed_2D_P3P2_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_momentum_system<VD_Mixed_2D_P3P3_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_momentum_system<VD_Mixed_2D_P4P3_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
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
        configure_momentum_system<VD_Mixed_3D_P1P1_Traits>(eq, mesh, a01, a10, a11, as,
                                                    eq_vloc, eq_lps,
                                                    eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        configure_momentum_system<VD_Mixed_3D_P2P1_Traits>(eq, mesh, a01, a10, a11, as,
                                                    eq_vloc, eq_lps,
                                                    eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        configure_momentum_system<VD_Mixed_3D_P3P2_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
        return;
      }
      if (pol_degree_Q == 3) {
        configure_momentum_system<VD_Mixed_3D_P3P3_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
        return;
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        configure_momentum_system<VD_Mixed_3D_P4P3_Traits>(
            eq, mesh, a01, a10, a11, as, eq_vloc, eq_lps, eq_lps_tensor, eq_h);
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

std::shared_ptr<FunctionSpace>
VD_NSEMomEquation::create_velocity_space(std::shared_ptr<const Mesh> mesh,
                                         Parameters &p) {
  const std::size_t D = mesh->topology().dim();
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];

  if (D == 2) {
    switch (pol_degree_V_M) {
    case 1:
      if (pol_degree_Q == 1) {
        return make_velocity_space<VD_Mixed_2D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_velocity_space<VD_Mixed_2D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_velocity_space<VD_Mixed_2D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_2D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_2D_P4P3_Traits>(mesh);
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
        return make_velocity_space<VD_Mixed_3D_P1P1_Traits>(mesh);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_velocity_space<VD_Mixed_3D_P2P1_Traits>(mesh);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_velocity_space<VD_Mixed_3D_P3P2_Traits>(mesh);
      }
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_3D_P3P3_Traits>(mesh);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_3D_P4P3_Traits>(mesh);
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

std::shared_ptr<FunctionSpace> VD_NSEMomEquation::create_velocity_space(
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
        return make_velocity_space<VD_Mixed_2D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_velocity_space<VD_Mixed_2D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_velocity_space<VD_Mixed_2D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_2D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_2D_P4P3_Traits>(mesh, periodic);
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
        return make_velocity_space<VD_Mixed_3D_P1P1_Traits>(mesh, periodic);
      }
      break;
    case 2:
      if (pol_degree_Q == 1) {
        return make_velocity_space<VD_Mixed_3D_P2P1_Traits>(mesh, periodic);
      }
      break;
    case 3:
      if (pol_degree_Q == 2) {
        return make_velocity_space<VD_Mixed_3D_P3P2_Traits>(mesh, periodic);
      }
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_3D_P3P3_Traits>(mesh, periodic);
      }
      break;
    case 4:
      if (pol_degree_Q == 3) {
        return make_velocity_space<VD_Mixed_3D_P4P3_Traits>(mesh, periodic);
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
