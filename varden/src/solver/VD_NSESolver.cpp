// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Copyright (c) 2025 Jonathan Wiskandt
// Licensed under the MIT License – see the LICENSE file for details

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip> // Include this header to use std::setprecision
#include <sstream>

#include <dolfin/common/Timer.h>
#include <dolfin/common/timing.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/io/File.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/refinement/refine.h>

///
namespace dolfin {
class Assembler;
class Constant;
template <typename T> class MeshFunction;
} // namespace dolfin

///
using dolfin::assemble;
using dolfin::CellIterator;
using dolfin::Constant;
using dolfin::error;
using dolfin::FunctionSpace;
using dolfin::info;
using dolfin::Parameters;
using dolfin::set_log_active;
using dolfin::tic;
using dolfin::toc;

#include "DofMappings.h"
#include "FileSaver.h"
#include "NodeNormal.h"
#include "Norms.h"
#include "PDESmooth.h"
#include "SlipBC.h"

#include "VD_NSEManifoldEquation.h"
#include "VD_NSEMomEquation.h"
#include "VD_NSEPressEquation.h"
#include "VD_NSESolver.h"
#include "VD_NSETracerEquation.h"

#include "ufl/VD_Mixed_2D_P1P1.h"
#include "ufl/VD_Mixed_3D_P1P1.h"

#include "ufl/VD_Tracer_2D_P1P1.h"
#include "ufl/VD_Tracer_3D_P1P1.h"

#include "ufl/troubled_cells_boundary/BoundaryMarker2D.h"
#include "ufl/troubled_cells_boundary/BoundaryMarker3D.h"

// ocean ////
#include "FarFieldTracer.h"
#include "FrictionVelocity.h"
#include "ThreeEquations.h"

using namespace fem_varden_flow;

//-----------------------------------------------------------------------------
VD_NSESolver::VD_NSESolver(
    std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
    std::shared_ptr<const Expression> initial_rho,
    std::shared_ptr<const Expression> initial_P,
    std::shared_ptr<const Expression> f,
    std::shared_ptr<const Expression> f_man,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
    std::shared_ptr<const std::vector<SlipBC *>> slip_bcs,
    std::shared_ptr<const Expression> slipbc_marker,
    std::shared_ptr<const Expression> slipbc_marker_fn, Parameters &p,
    double &time, double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  slip_bcs_ = slip_bcs;
  slipbc_marker_ = slipbc_marker;
  slipbc_marker_fn_ = slipbc_marker_fn;
  initial_P_ = initial_P;
  initial_rho_ = initial_rho;
  init();
}

//-----------------------------------------------------------------------------
VD_NSESolver::VD_NSESolver(std::shared_ptr<const Mesh> mesh,
                           Expression const &initial_u,
                           std::shared_ptr<const Expression> initial_rho,
                           std::shared_ptr<const Expression> initial_P,
                           std::shared_ptr<const Expression> f,
                           std::shared_ptr<const Expression> f_man,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
                           Parameters &p, double &time, double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  initial_P_ = initial_P;
  initial_rho_ = initial_rho;
  init();
}

//-----------------------------------------------------------------------------
VD_NSESolver::VD_NSESolver(std::shared_ptr<const Mesh> mesh,
                           Expression const &initial_u,
                           std::shared_ptr<const Expression> initial_rho,
                           std::shared_ptr<const Expression> initial_P,
                           std::shared_ptr<const Expression> f,
                           std::shared_ptr<const Expression> f_man,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
                           std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
                           std::shared_ptr<const FunctionSpace> V,
                           std::shared_ptr<const FunctionSpace> Q,
                           std::shared_ptr<const FunctionSpace> M,
                           Parameters &p, double &time, double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  V_ = V;
  Q_ = Q;
  M_ = M;
  initial_P_ = initial_P;
  initial_rho_ = initial_rho;
  init();
}

//-----------------------------------------------------------------------------
VD_NSESolver::VD_NSESolver(
    std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
    std::shared_ptr<const Expression> initial_rho,
    std::shared_ptr<const Expression> initial_P,
    std::shared_ptr<const Expression> f,
    std::shared_ptr<const Expression> f_man,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
    std::shared_ptr<const std::vector<SlipBC *>> slip_bcs,
    std::shared_ptr<const Expression> slipbc_marker,
    std::shared_ptr<const Expression> slipbc_marker_fn,
    std::shared_ptr<const FunctionSpace> V,
    std::shared_ptr<const FunctionSpace> Q,
    std::shared_ptr<const FunctionSpace> M, Parameters &p, double &time,
    double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  V_ = V;
  Q_ = Q;
  M_ = M;
  slip_bcs_ = slip_bcs;
  slipbc_marker_ = slipbc_marker;
  slipbc_marker_fn_ = slipbc_marker_fn;
  initial_P_ = initial_P;
  initial_rho_ = initial_rho;
  init();
}

VD_NSESolver::VD_NSESolver(
    std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
    std::shared_ptr<const Expression> initial_rho,
    std::shared_ptr<const Expression> initial_P,
    std::shared_ptr<const Expression> initial_T,
    std::shared_ptr<const Expression> f,
    std::shared_ptr<const Expression> f_man,
    std::shared_ptr<const Expression> f_T,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_T,
    std::shared_ptr<const std::vector<SlipBC *>> slip_bcs,
    std::shared_ptr<const Expression> slipbc_marker,
    std::shared_ptr<const Expression> slipbc_marker_fn,
    std::shared_ptr<const FunctionSpace> V,
    std::shared_ptr<const FunctionSpace> Q,
    std::shared_ptr<const FunctionSpace> M, Parameters &p, double &time,
    double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  V_ = V;
  Q_ = Q;
  M_ = M;
  slip_bcs_ = slip_bcs;
  slipbc_marker_ = slipbc_marker;
  slipbc_marker_fn_ = slipbc_marker_fn;
  initial_rho_ = initial_rho;
  initial_T_ = initial_T;
  f_T_ = f_T;
  bcs_T_ = bcs_T;
  initial_P_ = initial_P;
  constantTemperature = false;
  init();
  init_temperature();
}

VD_NSESolver::VD_NSESolver(
    std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
    std::shared_ptr<const Expression> initial_rho,
    std::shared_ptr<const Expression> initial_P,
    std::shared_ptr<const Expression> initial_T,
    std::shared_ptr<const Expression> initial_S,
    std::shared_ptr<const Expression> f,
    std::shared_ptr<const Expression> f_man,
    std::shared_ptr<const Expression> f_T,
    std::shared_ptr<const Expression> f_S,
    std::shared_ptr<const Expression> f_source_T,
    std::shared_ptr<const Expression> f_reaction_T,
    std::shared_ptr<const Expression> f_source_S,
    std::shared_ptr<const Expression> f_reaction_S,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_T,
    std::shared_ptr<std::vector<DirichletBC *>> bcs_S,
    std::shared_ptr<const std::vector<SlipBC *>> slip_bcs,
    std::shared_ptr<const Expression> slipbc_marker,
    std::shared_ptr<const Expression> slipbc_marker_fn,
    std::shared_ptr<const FunctionSpace> V,
    std::shared_ptr<const FunctionSpace> Q,
    std::shared_ptr<const FunctionSpace> M, Parameters &p,
    std::shared_ptr<MeshFunction<std::size_t>> sub_domains, double &time,
    double dt)
    : mesh_(mesh), initial_u(initial_u), f(f), f_man(f_man), bcs_u_(bcs_u),
      bcs_P_(bcs_P), bcs_rho_(bcs_rho), time_(time), dt_(dt) {
  parameters = p;
  V_ = V;
  Q_ = Q;
  M_ = M;
  slip_bcs_ = slip_bcs;
  slipbc_marker_ = slipbc_marker;
  slipbc_marker_fn_ = slipbc_marker_fn;
  sub_domains_ = sub_domains;
  initial_rho_ = initial_rho;
  initial_T_ = initial_T;
  initial_S_ = initial_S;
  initial_P_ = initial_P;
  f_T_ = f_T;
  f_source_T_ = f_source_T;
  f_reaction_T_ = f_reaction_T;
  f_source_S_ = f_source_S;
  f_reaction_S_ = f_reaction_S;
  f_S_ = f_S;
  bcs_T_ = bcs_T;
  bcs_S_ = bcs_S;
  constantTemperature = false;
  init();
  init_temperature();
  init_ocean();
}

//-----------------------------------------------------------------------------
void VD_NSESolver::init() {
  dt_pid_prev_error_ = 1.0;
  dt_pid_has_prev_error_ = false;
  constant_density = parameters["constant_density"];
  remove_pressure_average = parameters["remove_pressure_average"];
  remove_rho_average = parameters["remove_rho_average"];
  initial_guess_extrapolation = parameters["initial_guess_extrapolation"];
  bad_cells_boundary_condition = parameters["bad_cells_boundary_condition"];
  melt_with_virtual_flux = parameters["melt_with_virtual_flux"];
  std::string const demo_type = parameters["demo_type"];

  double solver_tolerance_LPS = parameters["solver_tolerance_LPS"];
  double solver_tolerance_tracer = parameters["solver_tolerance_tracer"];

  dim_ = mesh_->topology().dim();

  LPS_EQ_.ls = std::make_shared<LinearSolver>("cg", "jacobi");
  LPS_EQ_.ls->parameters["relative_tolerance"] = solver_tolerance_LPS;
  LPS_EQ_.ls->parameters["absolute_tolerance"] = 1.0e-20;
  LPS_EQ_.ls->parameters["nonzero_initial_guess"] = true;

  LPS_TENSOR_EQ_.ls = std::make_shared<LinearSolver>("gmres", "jacobi");
  LPS_TENSOR_EQ_.ls->parameters["relative_tolerance"] = solver_tolerance_LPS;
  LPS_TENSOR_EQ_.ls->parameters["absolute_tolerance"] = 1.0e-20;
  LPS_TENSOR_EQ_.ls->parameters["nonzero_initial_guess"] = true;

  if (V_ && Q_ && M_) {

    VD_NSETracerEquation::init(mesh_, M_, TRACER_EQ_, parameters);

    VD_NSEPressEquation::init(mesh_, Q_, PRESS_INITIAL_EQ_, OUTFLOW_EQ_,
                              parameters);

    VD_NSEMomEquation::init_mixed(mesh_, V_, Q_, MOM_EQ_, parameters, a01, a10,
                                  a11, aSchur, VLOC_EQ_, LPS_EQ_,
                                  LPS_TENSOR_EQ_, H_EQ_);
    VD_Divergence::init(mesh_, DIV_FORM_, parameters);
  }

  else {
    error("No function spaces provided.");
  }

  FUNC_ = std::make_shared<VD_Functionals>(mesh_, parameters);

  if (Q_->dim() != PRESS_INITIAL_EQ_.Q->dim() ||
      V_->dim() != MOM_EQ_.V->dim() || M_->dim() != TRACER_EQ_.M->dim())
    error("Wrong spaces!");

  double const sc = parameters["schmidt_number"];

  c_brezzi = parameters["C_brezzi"];

  scalar = std::make_shared<Scalar>(MPI_COMM_WORLD);
  Ck = std::make_shared<Constant>(1.0);
  Ck0 = std::make_shared<Constant>(1.0);
  Ck00 = std::make_shared<Constant>(1.0);
  Ck000 = std::make_shared<Constant>(1.0);
  C_BDF1_ = std::make_shared<Constant>(0.0);
  C_BDF3_ = std::make_shared<Constant>(0.0);
  C_BDF4_ = std::make_shared<Constant>(0.0);
  C_crank_nicolson_ = std::make_shared<Constant>(1.0);
  Crho_inf = std::make_shared<Constant>(1.0);
  Cvloc_inf = std::make_shared<Constant>(1.0);
  C_brezzi_ = std::make_shared<Constant>(c_brezzi);

  if (dim_ == 2) {
    e_x_ = std::make_shared<Constant>(1.0, 0.0);
    e_y_ = std::make_shared<Constant>(0.0, 1.0);
    e_z_ = std::make_shared<Constant>(0.0, 0.0);
  } else if (dim_ == 3) {
    e_x_ = std::make_shared<Constant>(1.0, 0.0, 0.0);
    e_y_ = std::make_shared<Constant>(0.0, 1.0, 0.0);
    e_z_ = std::make_shared<Constant>(0.0, 0.0, 1.0);
  }

  Frho_inf_K_ = std::make_shared<Function>(MOM_EQ_.DG0);
  Frho_inf_patch_ = std::make_shared<Function>(TRACER_EQ_.M);

  double const nu = parameters["nu"];
  Cnu_ = std::make_shared<Constant>(nu);

  double kappa;
  if (sc <= 0.0)
    error("Schmidt number must be positive");

  if (sc < 1.0e10) {
    kappa = nu / sc;
  } else {
    kappa = 0.0;
  }

  auto Czero = std::make_shared<Constant>(0.0);

  if (f_source_S_ == nullptr) {
    f_source_S_ = Czero;
    f_source_T_ = Czero;
    f_reaction_T_ = Czero;
    f_reaction_S_ = Czero;
  }

  marker_bad_cells_ =
      std::make_shared<dolfin::MeshFunction<std::size_t>>(mesh_, dim_, 0);

  C_kappa_ = std::make_shared<Constant>(kappa);
  Cgamma_grad_div = std::make_shared<Constant>(parameters["gamma_grad_div"]);
  C_ddn1_ = std::make_shared<Constant>(parameters["C_ddn_penalty1"]);
  C_ddn2_ = std::make_shared<Constant>(parameters["C_ddn_penalty2"]);

  C_skew_factor_ = std::make_shared<Constant>(parameters["skew_factor"]);

  C_skew_rho_factor_ =
      std::make_shared<Constant>(parameters["skew_rho_factor"]);

  Cslip_penalty = std::make_shared<Constant>(parameters["slip_penalty"]);
  Cslip_penalty_fn = std::make_shared<Constant>(parameters["slip_penalty_fn"]);
  Crv_smoothing_weight =
      std::make_shared<Constant>(parameters["rv_smoothing_weight"]);
  C_newton_ = std::make_shared<Constant>(0.0);
  int temp = parameters["project_uu"];
  project_uu = (bool)temp;
  C_project_uu_ = std::make_shared<Constant>((double)temp);

  C_shift_factor_ =
      std::make_shared<Constant>(parameters["emac_pressure_shift"]);

  C_max_ = std::make_shared<Constant>(parameters["C_max"]);
  C_max_vms_ = std::make_shared<Constant>(parameters["C_max_vms"]);

  if (demo_type == "ocean" && dim_ == 3) {
    C_omega_ = std::make_shared<Constant>(parameters["f_ocean_coriolis"]);
  } else {
    C_omega_ = std::make_shared<Constant>(0.0);
  }

  auto C_mesh_size_boundary_weight =
      std::make_shared<Constant>(parameters["mesh_size_boundary_weight"]);
  int directional_wave_speed = parameters["directional_wave_speed"];
  C_directional_wave_speed_ =
      std::make_shared<Constant>(directional_wave_speed);
  C_rho_mass_ = std::make_shared<Constant>(mass);
  C_T_mass_ = std::make_shared<Constant>(mass_T);
  C_S_mass_ = std::make_shared<Constant>(mass_S);
  C_kappa_T_ = std::make_shared<Constant>(0.0);
  C_zero_ = std::make_shared<Constant>(0.0);
  if (dim_ == 2) {
    C_zeros_ = std::make_shared<Constant>(0.0, 0.0);
  } else if (dim_ == 3) {
    C_zeros_ = std::make_shared<Constant>(0.0, 0.0, 0.0);
  }

  C_zero_ = std::make_shared<Constant>(0.0);
  C_one_ = std::make_shared<Constant>(1.0);

  rho_bar_ = std::make_shared<double>(1.0);
  T_bar_ = std::make_shared<double>(1.0);
  S_bar_ = std::make_shared<double>(1.0);

  C_eps_norm_ = std::make_shared<Constant>(parameters["res_scaling_C_l"]);

  C_slip_friction_ = std::make_shared<Constant>(0.0);
  C_nitsche_penalty_ = std::make_shared<Constant>(0.0);

  Fu_t_ = std::make_shared<Function>(MOM_EQ_.V);
  Fdtu_ = std::make_shared<Function>(MOM_EQ_.V);
  Fgrad_S_ = std::make_shared<Function>(
      LPS_EQ_.V); // I dont know if it matters if I use V_M or V
  Fgrad_T_ = std::make_shared<Function>(LPS_EQ_.V);
  Fgrad_rho_ = std::make_shared<Function>(LPS_EQ_.V);
  Fgrad_p_ = std::make_shared<Function>(LPS_EQ_.V);
  Fgrad_rho_->vector()->zero();
  Fgrad_p_->vector()->zero();
  Fgrad_u_ = std::make_shared<Function>(LPS_TENSOR_EQ_.V);
  F_p_div_u_ = std::make_shared<Function>(M_);
  Frho_t_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_exact_ = std::make_shared<Function>(TRACER_EQ_.M);
  Fvloc_ = std::make_shared<Function>(TRACER_EQ_.M);
  Fvloc_x_ = std::make_shared<Function>(TRACER_EQ_.M);
  Fvloc_y_ = std::make_shared<Function>(TRACER_EQ_.M);
  Fvloc_z_ = std::make_shared<Function>(TRACER_EQ_.M);
  FP_outflow_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  F_melting_flux_S_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_melting_flux_T_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_flux_T_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_flux_S_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_melt_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_gammaT_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_gammaS_ = std::make_shared<Function>(TRACER_EQ_.M);
  F_uu_projected_ = std::make_shared<Function>(TRACER_EQ_.M);

  Fu_boundary_ = std::make_shared<Function>(MOM_EQ_.V);

  Fu_np1_ = std::make_shared<Function>(MOM_EQ_.V);
  Fu_n_ = std::make_shared<Function>(MOM_EQ_.V);
  Fu_nm1_ = std::make_shared<Function>(MOM_EQ_.V);
  Fu_nm2_ = std::make_shared<Function>(MOM_EQ_.V);
  Fu_nm3_ = std::make_shared<Function>(MOM_EQ_.V);

  Fmom_ = std::make_shared<Function>(MOM_EQ_.V);

  Frho_np1_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_n_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_nm1_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_nm2_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_nm3_ = std::make_shared<Function>(TRACER_EQ_.M);
  Frho_bar_ = std::make_shared<Function>(MOM_EQ_.DG0);

  FT_np1_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_n_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_nm1_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_nm2_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_nm3_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_t_ = std::make_shared<Function>(TRACER_EQ_.M);
  FT_bar_ = std::make_shared<Function>(MOM_EQ_.DG0);

  FS_np1_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_n_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_nm1_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_nm2_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_nm3_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_t_ = std::make_shared<Function>(TRACER_EQ_.M);
  FS_bar_ = std::make_shared<Function>(MOM_EQ_.DG0);

  Fmom_K_ = std::make_shared<Function>(MOM_EQ_.V_DG0);

  FP_np1_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  FP_n_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  FP_nm1_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  FP_nm2_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  FP_nm3_ = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);

  Fdiv_ = std::make_shared<Function>(DIV_FORM_.M);
  Fgrad_phi_DG0_ = std::make_shared<Function>(MOM_EQ_.V_DG0);

  A00 = std::make_shared<Matrix>();
  A01 = std::make_shared<Matrix>();
  A10 = std::make_shared<Matrix>();
  A11 = std::make_shared<Matrix>();

  b0 = std::make_shared<Vector>();
  b1 = std::make_shared<Vector>();

  A_con_ = std::make_shared<Matrix>();
  A_press_ = std::make_shared<Matrix>();

  b_con_ = std::make_shared<Vector>();
  b_press_ = std::make_shared<Vector>();

  Fnormal_ = std::make_shared<Function>(MOM_EQ_.V);
  Ftau1_ = std::make_shared<Function>(MOM_EQ_.V);
  Ftau2_ = std::make_shared<Function>(MOM_EQ_.V);
  Fslipbc_marker_ = std::make_shared<Function>(TRACER_EQ_.M);
  Fslipbc_marker_fn_ = std::make_shared<Function>(TRACER_EQ_.M);
  if (slipbc_marker_ != nullptr) {
    Fslipbc_marker_->interpolate(*slipbc_marker_);
  }
  if (slipbc_marker_fn_ != nullptr) {
    Fslipbc_marker_fn_->interpolate(*slipbc_marker_fn_);
  }

  auto mesh_fine = *mesh_;
  refine(mesh_fine, mesh_fine, false);
  refine(mesh_fine, mesh_fine, false);
  auto mesh_fine_ = std::make_shared<Mesh>(mesh_fine);

  if (dim_ == 2) {
    V_P1_ =
        std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh_fine_);
  } else {
    V_P1_ =
        std::make_shared<VD_Mixed_3D_P1P1::CoefficientSpace_u_n>(mesh_fine_);
  }

  unit_node_normal_ = std::make_shared<fem_varden_flow::NodeNormal>(MOM_EQ_.V, V_P1_);
  *Fnormal_ = unit_node_normal_->normal();
  *Ftau1_ = unit_node_normal_->tau1();
  *Ftau2_ = unit_node_normal_->tau2();

  /// tabulate all dofs
  DofMappings dofmappings;
  dofmappings.get_all_dofs(*Fu_np1_, dof_V_);
  dofmappings.get_all_dofs(*FP_np1_, dof_Q_);
  dofmappings.get_all_dofs(*Frho_np1_, dof_M_);
  dofmappings.get_all_dofs(*Frho_inf_K_, dof_DG0_);
  dofmappings.get_all_dofs(*Fgrad_phi_DG0_, dof_DG0_V_);

  int const pol_degree = parameters["PolDegree_V_M"];

  Fh_PX_ = std::make_shared<Function>(MOM_EQ_.V);
  F_one_ = std::make_shared<Function>(TRACER_EQ_.M);
  *F_one_->vector() = 1.0;
  Matrix M_h;
  Vector b_h;

  H_EQ_.a->set_coefficient("weight", Crv_smoothing_weight);

  H_EQ_.a->set_coefficient("C_grad_phi_h", C_zero_);
  H_EQ_.a->set_coefficient("h_grad_phi", Fgrad_phi_DG0_);
  H_EQ_.a->set_coefficient("mesh_size_boundary_weight",
                           C_mesh_size_boundary_weight);
  H_EQ_.L->set_coefficient("mesh_size_boundary_weight",
                           C_mesh_size_boundary_weight);
  H_EQ_.L->set_coefficient("C_grad_phi_h", C_zero_);
  H_EQ_.L->set_coefficient("h_grad_phi", Fgrad_phi_DG0_);
  assemble(b_h, *H_EQ_.L);
  assemble(M_h, *H_EQ_.a);
  auto ls_temp = std::make_shared<LinearSolver>("cg", "jacobi");
  ls_temp->parameters["relative_tolerance"] = 1.0e-14;
  ls_temp->parameters["absolute_tolerance"] = 1.0e-20;
  ls_temp->parameters["nonzero_initial_guess"] = true;

  set_log_active(true);
  info("Solving for h...");

  int iterations_h = ls_temp->solve(M_h, *Fh_PX_->vector(), b_h);
  Fh_PX_->vector()->abs();
  info("Solved for h. Took " + std::to_string(iterations_h) + " iterations.");
  set_log_active(false);

  ls_mass_ = std::make_shared<LinearSolver>("cg", "jacobi");
  ls_mass_->parameters["relative_tolerance"] = solver_tolerance_tracer / 100.0;
  ls_mass_->parameters["absolute_tolerance"] = 1.0e-20;
  ls_mass_->parameters["nonzero_initial_guess"] = true;

  // Initialize nonlinear viscosity
  NV_ = std::make_shared<VD_NonlinearViscosity>(mesh_, TRACER_EQ_.M, dof_M_,
                                                dof_V_, parameters);
  NV_->Fu = Fu_np1_;
  NV_->Frho = Frho_np1_;
  NV_->FP = FP_np1_;
  NV_->Fvloc = Fvloc_;
  NV_->Fh_PX = Fh_PX_;
  NV_->Fres_u_fine_ = Fres_u_fine_;
  NV_->Fres_rho_fine_ = Fres_rho_fine_;
  NV_->Fres_T_fine_ = Fres_T_fine_;
  NV_->Fres_S_fine_ = Fres_S_fine_;
  NV_->FT = FT_np1_;
  NV_->FS = FS_np1_;
  NV_->coefficients_S_residual_ = coefficients_S_residual;
  NV_->coefficients_T_residual_ = coefficients_T;
  NV_->constantTemperature = constantTemperature;

  // Collect coefficients_rho_u for VD solver
  coefficients_rho_u = {{"u", Fu_np1_},
                        {"nu", Cnu_},
                        {"p", FP_np1_},
                        {"rho", Frho_np1_},
                        {"u_t", Fu_t_},
                        {"rho_t", Frho_t_},
                        {"f", f},
                        {"f_man", f_man},
                        {"emac_shift", C_shift_factor_},
                        {"eps_division", C_eps_norm_},
                        {"C_skew_rho_factor", C_skew_rho_factor_},
                        {"rho_K", Frho_bar_},
                        {"kappa", C_kappa_},
                        {"weight", Crv_smoothing_weight},
                        {"gamma_grad_div", Cgamma_grad_div},
                        {"C_directional_wave_speed", C_directional_wave_speed_},
                        {"C_nitsche_penalty", C_nitsche_penalty_},
                        {"C_omega", C_omega_},
                        {"h", Fh_PX_},
                        {"h_grad_phi", Fh_PX_},
                        {"slip_marker", Fslipbc_marker_},
                        {"slip_marker_fn", Fslipbc_marker_fn_},
                        {"slip_penalty", Cslip_penalty},
                        {"slip_penalty_fn", Cslip_penalty_fn},
                        {"node_normal", Fnormal_},
                        {"rho_n", Frho_n_},
                        {"u_n", Fu_n_},
                        {"rho_np1", Frho_np1_},
                        {"p_n", FP_n_},
                        {"k", Ck},
                        {"k0", Ck0},
                        {"k00", Ck00},
                        {"k000", Ck000},
                        {"u_N", Fu_np1_},
                        {"CN", C_newton_},
                        {"C_project_uu", C_project_uu_},
                        {"uu_projected", F_uu_projected_},
                        {"rho_bar", C_rho_mass_},
                        {"mom_bar", Fmom_K_},
                        {"C_skew_factor", C_skew_factor_},
                        {"C_brezzi", C_brezzi_},
                        {"u_nm1", Fu_nm1_},
                        {"u_nm2", Fu_nm2_},
                        {"u_nm3", Fu_nm3_},
                        {"rho_nm1", Frho_nm1_},
                        {"rho_nm2", Frho_nm2_},
                        {"rho_nm3", Frho_nm3_},
                        {"u_nm2", Fu_nm2_},
                        {"f_source", Czero},
                        {"f_reaction", Czero},
                        {"C_ddn1", C_ddn1_},
                        {"C_ddn2", C_ddn2_},
                        {"boundary_function", FP_outflow_}, // initial_P_
                        {"neumann_flux", F_melting_flux_T_},
                        {"C_slip_friction", C_slip_friction_},
                        {"node_tangent1", Ftau1_},
                        {"p_grad_rho", Fgrad_rho_},
                        {"p_grad_p", Fgrad_p_},
                        {"p_grad_u", Fgrad_u_},
                        {"p_div_u", Fdiv_},
                        {"C_max", C_max_},
                        {"C_max_vms", C_max_vms_},
                        {"res_indicator", NV_->get_res_rho()},
                        {"res_indicator_rho", NV_->get_res_rho()},
                        {"res_indicator_u", NV_->get_res_u()},
                        {"vloc", Fvloc_},
                        {"vloc_x", Fvloc_x_},
                        {"vloc_y", Fvloc_y_},
                        {"vloc_z", Fvloc_z_},
                        {"C_crank_nicolson", C_crank_nicolson_},
                        {"C_BDF1", C_BDF1_},
                        {"C_BDF3", C_BDF3_},
                        {"C_BDF4", C_BDF4_},
                        {"S_bar", C_S_mass_},
                        {"T_bar", C_T_mass_},
                        {"g", initial_rho_}};

  coefficients_only_P = {{"rho_n", FP_n_},
                         {"rho_np1", FP_np1_},
                         {"rho_nm1", FP_nm1_},
                         {"rho_nm2", FP_nm2_},
                         {"rho_nm3", FP_nm3_}};

  if (!f_S_) {
    if (dim_ == 2) {
      f_S_ = std::make_shared<Constant>(0.0, 0.0);
    } else {
      f_S_ = std::make_shared<Constant>(0.0, 0.0, 0.0);
    }
  }

  if (!f_T_) {
    if (dim_ == 2) {
      f_T_ = std::make_shared<Constant>(0.0, 0.0);
    } else {
      f_T_ = std::make_shared<Constant>(0.0, 0.0, 0.0);
    }
  }
  // Collect coefficients for temperature that is needed everywhere else...
  coefficients_T = {{"T_t", FT_t_},
                    {"T", FT_np1_},
                    {"f_T", f_T_},
                    {"f_source", f_source_T_},
                    {"f_reaction", f_reaction_T_},
                    {"T_n", FT_n_},
                    {"T_np1", FT_np1_},
                    {"T_nm1", FT_nm1_},
                    {"T_nm2", FT_nm2_},
                    {"T_nm3", FT_nm3_},
                    {"kappa_T", C_kappa_T_},
                    {"T_F", FT_np1_}};

  coefficients_S_residual = {{"T_t", FS_t_},
                             {"T", FS_np1_},
                             {"T_n", FS_n_},
                             {"T_np1", FS_np1_},
                             {"T_nm1", FS_nm1_},
                             {"T_nm2", FS_nm2_},
                             {"T_nm3", FS_nm3_},
                             {"kappa_T", C_kappa_S_},
                             {"f_source", f_source_S_},
                             {"f_reaction", f_reaction_S_},
                             {"T_F", FT_np1_}};

  coefficients_S = {{"S_t", FS_t_},
                    {"S", FS_np1_},
                    {"f_S", f_S_},
                    {"f_source", f_source_S_},
                    {"f_reaction", f_reaction_S_},
                    {"S_n", FS_n_},
                    {"S_np1", FS_np1_},
                    {"S_nm1", FS_nm1_},
                    {"S_nm2", FS_nm2_},
                    {"S_nm3", FS_nm3_},
                    {"kappa_S", C_kappa_S_},
                    {"S_F", FS_np1_}};

  LPS_TENSOR_EQ_.a->set_some_coefficients(coefficients_rho_u);
  LPS_EQ_.a->set_some_coefficients(coefficients_rho_u);
  assemble(LPS_EQ_.A, *LPS_EQ_.a);
  assemble(LPS_TENSOR_EQ_.A, *LPS_TENSOR_EQ_.a);

  NV_->RES.L->set_some_coefficients(coefficients_rho_u);
  NV_->RES.L->set_some_coefficients(coefficients_T);
  NV_->RES.L->set_some_coefficients(coefficients_S);
  NV_->RES.a->set_some_coefficients(coefficients_rho_u);

  NV_->assemble_system();

  MOM_EQ_.a->set_some_coefficients(coefficients_rho_u);
  MOM_EQ_.L->set_some_coefficients(coefficients_rho_u);
  MOM_EQ_.a->set_some_coefficients(coefficients_T);
  MOM_EQ_.L->set_some_coefficients(coefficients_T);

  MOM_EQ_.a->set_some_coefficients(coefficients_S);
  MOM_EQ_.L->set_some_coefficients(coefficients_S);

  TRACER_EQ_.a->set_some_coefficients(coefficients_rho_u);

  TRACER_EQ_.L->set_some_coefficients(coefficients_rho_u);

  VLOC_EQ_.a->set_some_coefficients(coefficients_rho_u);
  VLOC_EQ_.L->set_some_coefficients(coefficients_rho_u);
  VLOC_EQ_.a->set_coefficient("C_project_uu", C_zero_);
  VLOC_EQ_.L->set_coefficient("C_project_uu", C_zero_);

  assemble(VLOC_EQ_.A, *VLOC_EQ_.a);

  PRESS_INITIAL_EQ_.L->set_coefficient("f", f);
  PRESS_INITIAL_EQ_.L->set_coefficient("f_man", f_man);
  PRESS_INITIAL_EQ_.L->set_coefficient("u_np1", Fu_np1_);
  PRESS_INITIAL_EQ_.L->set_coefficient("rho_np1", Frho_np1_);
  PRESS_INITIAL_EQ_.L->set_coefficient("nu", Cnu_);

  TRACER_EQ_.L->ds = sub_domains_;
  TRACER_EQ_.a->ds = sub_domains_;

  MOM_EQ_.a->ds = sub_domains_;
  MOM_EQ_.a->dx = marker_bad_cells_;
  TRACER_EQ_.a->dx = marker_bad_cells_;
  TRACER_EQ_.L->dx = marker_bad_cells_;
  MOM_EQ_.L->ds = sub_domains_;
  a01->ds = sub_domains_;
  FUNC_->melt_rate->ds = sub_domains_;
  FUNC_->tau0->ds = sub_domains_;
  FUNC_->tau0->dS = sub_domains_;
  FUNC_->tau0->dx = sub_domains_;
  FUNC_->tau0->set_coefficient("C_volume", C_volume_);
  FUNC_->tau0->set_coefficient("u", Fu_np1_);

  ////////////////////////////////////////////////////////////////////////////

  double const tolerance = parameters["solver_tolerance"];
  double const tolerance_tracer = parameters["solver_tolerance_tracer"];
  int const max_iter = parameters["max_solver_iterations"];
  lsrho_ = std::make_shared<LinearSolver>("gmres", "jacobi");
  lsrho_->parameters["nonzero_initial_guess"] = true;
  lsrho_->parameters["maximum_iterations"] = max_iter;
  lsrho_->parameters["absolute_tolerance"] = 1e-20;
  lsrho_->parameters["relative_tolerance"] = tolerance_tracer;
  // lsrho_->parameters["monitor_convergence"] = true;

  lsP_ = std::make_shared<LinearSolver>("bicgstab", "hypre_amg");

  lsP_->parameters["relative_tolerance"] = tolerance;
  lsP_->parameters["absolute_tolerance"] = 1e-20;
  lsP_->parameters["nonzero_initial_guess"] = true;

  lsP_->parameters["maximum_iterations"] = max_iter;
  // lsP_->parameters["monitor_convergence"] = true;
  lsP_->parameters["divergence_limit"] = 1.0e15;
  lsrho_->parameters["divergence_limit"] = 1.0e15;
  // lsrho_->parameters["monitor_convergence"] = true;
  {

    PetscErrorCode ierr;

    a11->set_some_coefficients(coefficients_rho_u);

    ASchur = std::make_shared<Matrix>();
    double const c_laplace = parameters["C_laplace"];
    double const c_mass = parameters["C_mass"];
    C_laplace = std::make_shared<Constant>(c_laplace);
    C_mass = std::make_shared<Constant>(c_mass);
    aSchur->set_coefficient("C_laplace", C_laplace);
    aSchur->set_coefficient("C_mass", C_mass);
    aSchur->set_coefficient("rho_np1", Frho_np1_);
    assemble(*ASchur, *aSchur);

    MOM_EQ_.a->set_coefficient("rho_bar", C_rho_mass_);
    MOM_EQ_.a->set_some_coefficients(coefficients_rho_u);
    a01->set_some_coefficients(coefficients_rho_u);
    a11->set_some_coefficients(coefficients_rho_u);
    a10->set_some_coefficients(coefficients_rho_u);
    assemble(*A00, *MOM_EQ_.a);
    assemble(*A10, *a10);
    assemble(*A01, *a01);
    assemble(*A11, *a11);
    assemble(*b0, *MOM_EQ_.L);
    assemble(*b1,
             *PRESS_INITIAL_EQ_.L); // FIXME: Bug bug bug. But it is zeroed out
                                    // later so it doesnt matter anyway...

    M_L = std::make_shared<Matrix>(MPI_COMM_WORLD);
    assemble(*M_L, *a11);
    auto M_I = std::make_shared<Matrix>(MPI_COMM_WORLD);
    assemble(*M_I, *a11);
    get_dt_cfl(*Fvloc_, dof_M_, time_, dt_, parameters);

    // apply Dirichlet bc
    for (std::size_t i = 0; i < bcs_u_->size(); ++i) {
      (*bcs_u_)[i]->apply(*A00, *b0);
      (*bcs_u_)[i]->zero(*A01);
    }

    std::vector<Mat> petsc_mats(4);
    petsc_mats[0] = (as_type<PETScMatrix>(*A00).mat());
    petsc_mats[1] = (as_type<PETScMatrix>(*A01).mat());
    petsc_mats[2] = (as_type<PETScMatrix>(*A10).mat());
    petsc_mats[3] = (as_type<PETScMatrix>(*A11).mat());

    std::vector<IS> is(2);
    std::vector<IS> is_col(2);
    Mat A_nest_mat;
    ierr = MatCreateNest(MPI_COMM_WORLD, 2, nullptr, 2, nullptr,
                         petsc_mats.data(), &A_nest_mat);

    if (ierr != 0)
      error("Petsc error code %d when performing MatCreateNest", ierr);

    ierr = MatNestGetISs(A_nest_mat, is.data(), is_col.data());
    if (ierr != 0) {
      error("Petsc error code %d when performing MatNestGetISs", ierr);
    }

    std::vector<Vec> nest_in(2);
    std::vector<Vec> nest_in_2(2);
    Vec nest_rhs_out;
    Vec nest_sol_out;

    nest_in[0] = as_type<PETScVector>(*b0).vec();
    nest_in[1] = as_type<PETScVector>(*b1).vec();
    ierr = VecCreateNest(MPI_COMM_WORLD, 2, is.data(), nest_in.data(),
                         &nest_rhs_out);

    if (ierr != 0)
      error("Petsc error code %d when performing VecCreateNest", ierr);
    rhs_vec = std::make_shared<PETScVector>(nest_rhs_out);

    nest_in_2[0] = as_type<PETScVector>(*Fu_np1_->vector()).vec();
    nest_in_2[1] = as_type<PETScVector>(*FP_np1_->vector()).vec();
    ierr = VecCreateNest(MPI_COMM_WORLD, 2, is.data(), nest_in_2.data(),
                         &nest_sol_out);
    if (ierr != 0)
      error("Petsc error code %d when performing VecCreateNest", ierr);
    sol_vec = std::make_shared<PETScVector>(nest_sol_out);

    double const field0_tol = parameters["field0_tolerance"];
    double const field1_tol = parameters["field1_tolerance"];
    int const monitor_ksp = parameters["monitor_ksp"];
    int const field1_max_iter = parameters["field1_max_iter"];
    int const field0_max_iter = parameters["field0_max_iter"];
    int const user_schur = parameters["user_schur"];
    std::string const field0_ksp = parameters["field0_ksp"];
    std::string const field0_pc = parameters["field0_pc"];
    std::string const field1_ksp = parameters["field1_ksp"];
    std::string const field1_pc = parameters["field1_pc"];
    std::string const schur_type = parameters["schur_type"];
    std::string const upper = parameters["upper_pc"];
    std::string const lower = parameters["inner_pc"];

    PETScOptions::set("pc_type", "fieldsplit");
    PETScOptions::set("pc_fieldsplit_type", "schur");
    PETScOptions::set("pc_fieldsplit_schur_fact_type", schur_type);

    /////////////////////////////////////////////////////////////
    PETScOptions::set("fieldsplit_0_ksp_type", field0_ksp);
    PETScOptions::set("fieldsplit_0_pc_type", field0_pc);
    PETScOptions::set("fieldsplit_0_ksp_rtol", field0_tol);
    PETScOptions::set("fieldsplit_0_ksp_max_it", field0_max_iter);

    PETScOptions::set("fieldsplit_1_ksp_type", field1_ksp);
    PETScOptions::set("fieldsplit_1_pc_type", field1_pc);
    PETScOptions::set("fieldsplit_1_ksp_rtol", field1_tol);
    PETScOptions::set("fieldsplit_1_ksp_max_it", field1_max_iter);

    if (user_schur == 0) {
      PETScOptions::set("fieldsplit_1_pc_jacobi_type", "rowsum");

      PETScOptions::set("fieldsplit_1_inner_pc_jacobi_type", "rowsum");
      PETScOptions::set("fieldsplit_1_upper_pc_jacobi_type", "rowsum");

      // Speed up solution process. Only use preconditioner for some of the
      // steps.
      PETScOptions::set("fieldsplit_1_inner_ksp_type", "preonly");
      PETScOptions::set("fieldsplit_1_inner_pc_type", upper);
      PETScOptions::set("fieldsplit_1_upper_ksp_type", "preonly");
      PETScOptions::set("fieldsplit_1_upper_pc_type", lower);

      // Preconditioner for the schur solve. Can try a11 (default), user and
      // self (self doesnt work unless some conditions are met)
      PETScOptions::set("pc_fieldsplit_schur_precondition", "user");

      // what type of approximation of the schur complement to use. Default is
      // inv(diag(A11)). lump, blockdiag, diag This option doesn't seem to
      // change much.
      PETScOptions::set("fieldsplit_1_mat_schur_complement_ainv_type",
                        "blockdiag");
    }

    if (monitor_ksp == 1) {
      PETScOptions::set("fieldsplit_0_ksp_converged_reason");
      PETScOptions::set("fieldsplit_1_ksp_converged_reason");

      PETScOptions::set("ksp_view");
      PETScOptions::set("pc_view");
      PETScOptions::set("ksp_monitor_true_residual");
      PETScOptions::set("log_view"); // This will print detailed solver logs
    }

    // From Creff, Guermond 2024 (no difference)
    //  PETScOptions::set("-fieldsplit_1_pc_hypre_type", "boomeramg");
    //  PETScOptions::set("-fieldsplit_1_pc_hypre_boomeramg_strong_threshold",
    //  0.2); // or 0.7
    //  PETScOptions::set("-fieldsplit_1_pc_hypre_boomeramg_coarsen_type",
    //  "Falgout");
    //  PETScOptions::set("-fieldsplit_1_pc_hypre_boomeramg_relax_type_all",
    //  "Chebyshev");
    //  PETScOptions::set("-fieldsplit_1_pc_hypre_boomeramg_relax_max_iter", 1);
    //  // or 2

    schur_krylovsolver = std::make_shared<PETScKrylovSolver>("gmres", "none");

    PETScOptions::set("ksp_type", "fgmres");
    A_nest = std::make_shared<PETScMatrix>(A_nest_mat);

    PETScOptions::set("ksp_gmres_preallocate", false);

    schur_krylovsolver->set_from_options();

    KSPSetOperators(schur_krylovsolver->ksp(), A_nest_mat, A_nest_mat);

    PC pc;
    ierr = KSPGetPC(schur_krylovsolver->ksp(), &pc);
    if (ierr != 0) {
      error("Petsc error code %d when performing KSPGetPC", ierr);
    }

    ierr = PCFieldSplitSetIS(pc, "0", is[0]);
    if (ierr != 0) {
      error("Petsc error code %d when performing PCFieldSplitSetIS", ierr);
    }

    ierr = PCFieldSplitSetIS(pc, "1", is[1]);
    if (ierr != 0) {
      error("Petsc error code %d when performing PCFieldSplitSetIS", ierr);
    }

    if (user_schur == 0) {
      ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER,
                                     as_type<PETScMatrix>(*ASchur).mat());
      if (ierr != 0) {
        error("Petsc error code %d when performing PCFieldSplitSetSchurPreC",
              ierr);
      }
    }

    if (user_schur == 1) {
      // Supplying own schur
      ierr = PCSetUp(pc);
      KSP *subksp;
      PetscInt index_field1 = 1;
      if (ierr != 0) {
        error("Petsc error code %d when performing PCFieldSplitSetIS", ierr);
      }

      ierr = PCFieldSplitGetSubKSP(pc, &index_field1, &subksp);
      if (ierr != 0) {
        error("Petsc error code %d when performing PCFieldSplitSetIS", ierr);
      }

      ierr = KSPSetOperators(subksp[1], as_type<PETScMatrix>(*ASchur).mat(),
                             as_type<PETScMatrix>(*ASchur).mat());
      if (ierr != 0) {
        error("Petsc error code %d when performing PCFieldSplitSetIS", ierr);
      }
    }

    double const tolerance = parameters["solver_tolerance"];
    schur_krylovsolver->parameters["relative_tolerance"] = tolerance;
    schur_krylovsolver->parameters["absolute_tolerance"] = 1.0e-20;
    schur_krylovsolver->parameters["divergence_limit"] = 1.0e15;
    schur_krylovsolver->parameters["nonzero_initial_guess"] = true;
    schur_krylovsolver->parameters["maximum_iterations"] = max_iter;
  }

  // Divergence forms
  DIV_FORM_.L->set_coefficient("u", Fu_np1_);
  FUNC_->momentum_->set_coefficient("u", Fu_np1_);
  FUNC_->momentum_->set_coefficient("rho", Frho_np1_);

  FUNC_->angular_momentum_z->set_coefficient("u", Fu_np1_);
  FUNC_->angular_momentum_z->set_coefficient("rho", Frho_np1_);

  FUNC_->mass->set_coefficient("rho", Frho_np1_);

  FUNC_->L2rho->set_coefficient("rho", Frho_np1_);

  // for visualization of P2 and P3 solution:
  int const p3_visualization = parameters["P3_visualization"];
  mesh_fine_ = std::make_shared<Mesh>(*mesh_);
  if (pol_degree > 1) {
    bool redistribute = false;
    refine(*mesh_fine_, *mesh_, redistribute);
    if (pol_degree > 2 && p3_visualization == 1)
      refine(*mesh_fine_, *mesh_fine_, redistribute);
  }

  if (dim_ == 2) {
    V_fine =
        std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh_fine_);
    M_fine = std::make_shared<VD_Tracer_2D_P1P1::FunctionSpace>(mesh_fine_);
    Q_CG1_ = std::make_shared<VD_Tracer_2D_P1P1::FunctionSpace>(mesh_);
  } else if (dim_ == 3) {
    V_fine =
        std::make_shared<VD_Mixed_3D_P1P1::CoefficientSpace_u_n>(mesh_fine_);
    M_fine = std::make_shared<VD_Tracer_3D_P1P1::FunctionSpace>(mesh_fine_);
    Q_CG1_ = std::make_shared<VD_Tracer_3D_P1P1::FunctionSpace>(mesh_);
  }

  Fu_fine_ = std::make_shared<Function>(V_fine);
  Frho_fine_ = std::make_shared<Function>(M_fine);
  Fdiv_fine_ = std::make_shared<Function>(M_fine);
  Fmu_fine_ = std::make_shared<Function>(M_fine);
  Fres_u_fine_ = std::make_shared<Function>(M_fine);
  Fres_rho_fine_ = std::make_shared<Function>(M_fine);
  Fres_T_fine_ = std::make_shared<Function>(M_fine);

  NV_->Fres_u_fine_ = Fres_u_fine_;
  NV_->Fres_T_fine_ = Fres_T_fine_;
  NV_->Fres_S_fine_ = Fres_S_fine_;
  NV_->Fres_rho_fine_ = Fres_rho_fine_;

  file_saver_ = std::make_shared<FileSaver>(time_, parameters);
  file_saver_fine_ = std::make_shared<FileSaver>(time_, parameters);

  std::ostringstream stime;
  stime << "_t=" << time_;

  // file_saver_->add(*FP_np1_, "P" + stime.str());
  // file_saver_->add(*FP_outflow_, "P_outflow" + stime.str());

  Function initial_P(TRACER_EQ_.M);
  initial_P.interpolate(*initial_P_);
  // file_saver_->add(initial_P, "P_initial" + stime.str());

  // file_saver_->add_checkpoint(*Frho_np1_, "rho");
  file_saver_->add_checkpoint(*Fu_np1_, "u");
  // file_saver_->add(*Ftau1_, "tau1" + stime.str());
  // file_saver_->add(*Fnormal_, "node_normal" + stime.str());
  // file_saver_->add(*Fmom_K_, "mom_bar" + stime.str());
  // file_saver_->add(*Frho_inf_K_, "rho_K" + stime.str());

  // file_saver_->add(*Fu_boundary_, "u_boundary" + stime.str());
  // file_saver_->add(*Fslipbc_marker_, "slipbc_marker");
  // file_saver_->add(*Fvloc_, "Fvloc");

  // file_saver_->add(*Fdiv_, "div" + stime.str());
  // file_saver_->add(*NV_->Fnorm_u, "Norm_u" + stime.str());
  // file_saver_->add(*NV_->Fnorm_rho, "Norm_rho" + stime.str());
  // file_saver_->add(*NV_->Fnorm_mom, "Norm_mom" + stime.str());
  // file_saver_->add(*Frho_bar_, "rho_bar" + stime.str());
  // file_saver_->add(*Frho_inf_patch_, "rho_patch" + stime.str());

  file_saver_fine_->add(*Fu_np1_, V_fine, "u_fine" + stime.str());
  // file_saver_fine_->add(*Frho_np1_, M_fine, "rho_fine" + stime.str());
  // file_saver_fine_->add(*FP_np1_, M_fine, "p_fine" + stime.str());
  file_saver_fine_->add(*NV_->get_res_u(), M_fine, "res_u_fine" + stime.str());
  file_saver_fine_->add(*NV_->get_res_T(), M_fine, "res_T_fine" + stime.str());
  file_saver_fine_->add(*NV_->get_res_S(), M_fine, "res_S_fine" + stime.str());
  // file_saver_fine_->add(*NV_->get_res_rho(), M_fine,
  // "res_rho_fine" + stime.str());
  // // file_saver_fine_->add(*NV_->get_mu_h(), M_fine, "mu_fine" +
  // stime.str());
  // // file_saver_fine_->add(*NV_->get_kappa_h(), M_fine, "kappa_fine" +
  // stime.str());
  // // file_saver_fine_->add(*NV_->get_kappa_T_h(), M_fine, "kappa_T_fine" +
  // stime.str());
  // // file_saver_fine_->add(*Fdiv_, M_fine, "div_fine" + stime.str());
  // // file_saver_fine_->add(*Fvloc_, M_fine, "vloc_fine" + stime.str());
  // // file_saver_fine_->add(*Fvloc_x_, M_fine, "vloc_x_fine" + stime.str());
  // // file_saver_fine_->add(*Fvloc_y_, M_fine, "vloc_y_fine" + stime.str());
  // // file_saver_fine_->add(*Fvloc_z_, M_fine, "vloc_z_fine" + stime.str());

  /// volume of the mesh:
  mesh_volume_ = volume(*mesh_);

  functional_offset_ = assemble_functional(*(FUNC_->offset), *scalar);

  boundary_surface_ = 1.0;

  auto Fgrad_ = std::make_shared<Function>(TRACER_EQ_.M);
  Function Flump(M_fine);
  NV_->compute_grad_phi_CG1(Q_CG1_, *Fgrad_phi_DG0_);

  if (directional_wave_speed == 1) {
    H_EQ_.a->set_coefficient("C_grad_phi_h", C_one_);
    H_EQ_.a->set_coefficient("h_grad_phi", Fgrad_phi_DG0_);
    H_EQ_.L->set_coefficient("C_grad_phi_h", C_one_);
    H_EQ_.L->set_coefficient("h_grad_phi", Fgrad_phi_DG0_);
    assemble(b_h, *H_EQ_.L);
    assemble(M_h, *H_EQ_.a);
    set_log_active(true);
    info("Solving for h...");
    int iterations_h = ls_temp->solve(M_h, *Fh_PX_->vector(), b_h);
    Fh_PX_->vector()->abs();
    info("Solved for h. Took " + std::to_string(iterations_h) + " iterations.");
    set_log_active(false);
  }

  File file_mesh_size("results/pvd_files/mesh_size.pvd");
  file_mesh_size.write(*Fh_PX_);

  if (bad_cells_boundary_condition) {
    set_log_active(true);
    info("Identifying bad cells...");
    set_log_active(false);
    add_noslip_on_bad_cells();
    set_log_active(true);
    info("Finished identifying bad cells.");
    set_log_active(false);
  }

  set_log_active(true);
  info("Finished initialization with %i nodes", Frho_np1_->vector()->size());
  set_log_active(false);
}

//-----------------------------------------------------------------------------
void VD_NSESolver::init_temperature() {

  C_kappa_T_ = std::make_shared<Constant>(0.0);

  C_kappa_S_ = std::make_shared<Constant>(0.0);

  double kappa_T;
  double kappa_S;

  if (parameters.has_parameter("kappa_T")) {
    kappa_T = parameters["kappa_T"];
    kappa_S = parameters["kappa_S"];

  } else {
    // Create a dummy error message
    std::string error_message = "No value for kappa_T given in input file.";

    // Raise the error and print the error message
    dolfin_error("solver", "init_temperature", error_message.c_str());
  }

  *C_kappa_T_ = kappa_T;
  *C_kappa_S_ = kappa_S;

  C_skew_T_factor_ = std::make_shared<Constant>(parameters["skew_rho_factor"]);
  C_skew_S_factor_ = std::make_shared<Constant>(
      parameters["skew_rho_factor"]); // use same factor for both

  // Collect coefficients only used for temperature update
  coefficients_only_T = {{"kappa", C_kappa_T_},
                         {"res_indicator", NV_->get_res_T()},
                         {"rho_n", FT_n_},
                         {"rho_np1", FT_np1_},
                         {"rho_bar", C_T_mass_},
                         {"rho_nm1", FT_nm1_},
                         {"rho_nm2", FT_nm2_},
                         {"rho_nm3", FT_nm3_},
                         {"C_skew_rho_factor", C_skew_T_factor_},
                         {"f_source", f_source_T_},
                         {"f_reaction", f_reaction_T_},
                         {"neumann_flux", F_melting_flux_T_},
                         {"p_grad_rho", Fgrad_T_},
                         {"g", initial_T_}};

  coefficients_only_S = {{"kappa", C_kappa_S_},
                         {"res_indicator", NV_->get_res_S()},
                         {"rho_n", FS_n_},
                         {"rho_np1", FS_np1_},
                         {"rho_bar", C_S_mass_},
                         {"rho_nm1", FS_nm1_},
                         {"rho_nm2", FS_nm2_},
                         {"rho_nm3", FS_nm3_},
                         {"C_skew_rho_factor", C_skew_S_factor_},
                         {"f_source", f_source_S_},
                         {"f_reaction", f_reaction_S_},
                         {"neumann_flux", F_melting_flux_S_},
                         {"p_grad_rho", Fgrad_S_},
                         {"g", initial_S_}};

  TRACER_EQ_.a->set_some_coefficients(coefficients_only_T);
  TRACER_EQ_.L->set_some_coefficients(coefficients_only_T);

  FT_fine_ = std::make_shared<Function>(M_fine);
  FS_fine_ = std::make_shared<Function>(M_fine);

  std::ostringstream stime;
  stime << "_t=" << time_;
  file_saver_->add_checkpoint(*FT_np1_, "T");
  file_saver_fine_->add(*FT_np1_, M_fine, "T_fine" + stime.str());

  file_saver_->add_checkpoint(*FS_np1_, "S");

  file_saver_fine_->add(*FS_np1_, M_fine, "S_fine" + stime.str());
}

void VD_NSESolver::init_ocean() {
  // Create boundary mesh
  auto boundary_mesh_ = std::make_shared<BoundaryMesh>(*mesh_, "exterior");
  boundary_mesh_->bounding_box_tree(); // neccessary for eval to work

  VD_NSEManifoldEquation::init(boundary_mesh_, MANIFOLD_EQ_.M, MANIFOLD_EQ_,
                               parameters);

  Frho_boussinesq_ = std::make_shared<Function>(MOM_EQ_.M);
  Fmelt_avg_ = std::make_shared<Function>(MOM_EQ_.M);
  Frho_b_avg_ = std::make_shared<Function>(MOM_EQ_.M);
  Fu_avg_ = std::make_shared<Function>(MOM_EQ_.V);
  FT_avg_ = std::make_shared<Function>(MOM_EQ_.M);
  FS_avg_ = std::make_shared<Function>(MOM_EQ_.M);
  C_volume_ = std::make_shared<Constant>(1.0);

  if (melt_with_virtual_flux == 0 || melt_with_virtual_flux == 2) {
    F_melt_V_ = std::make_shared<Function>(V_->sub(0)->collapse());
    F_melt_VV_ = std::make_shared<Function>(V_);

    melt_vector_bc = std::make_shared<DirichletBC>(V_->sub(0)->collapse(),
                                                   F_melt_V_, sub_domains_, 1);
    melt_vector_bc0 =
        std::make_shared<DirichletBC>(V_->sub(0), F_melt_V_, sub_domains_, 1);
    melt_vector_bc1 =
        std::make_shared<DirichletBC>(V_->sub(1), F_melt_V_, sub_domains_, 1);
    if (dim_ == 3) {
      melt_vector_bc2 =
          std::make_shared<DirichletBC>(V_->sub(2), F_melt_V_, sub_domains_, 1);
    }

    if (melt_with_virtual_flux == 0) {
      FT_b_ = std::make_shared<Function>(MOM_EQ_.M);
      FS_b_ = std::make_shared<Function>(MOM_EQ_.M);

      melt_DBC_T_ =
          std::make_shared<DirichletBC>(MOM_EQ_.M, FT_b_, sub_domains_, 1);
      melt_DBC_S_ =
          std::make_shared<DirichletBC>(MOM_EQ_.M, FS_b_, sub_domains_, 1);
    }

    inflow_bc_melt_ =
        std::make_shared<SlipBC>(MOM_EQ_.V, V_P1_, sub_domains_, 1);
  }

  double solver_tolerance_tracer = parameters["solver_tolerance_tracer"];
  double restoring_region_width = parameters["restoring_region_width"];
  if (restoring_region_width <= 0.0) {
    *C_nitsche_penalty_ = parameters["C_nitsche_penalty"];
  } else {
    *C_nitsche_penalty_ = 0.0;
  }
  melting_with_rv = parameters["melting_with_rv"];
  Cmelt_smoothing_weight_ =
      std::make_shared<Constant>(parameters["melt_smoothing_weight"]);
  double c_friction = parameters["C_slip_friction"];
  *C_slip_friction_ = c_friction;

  if (melting_with_rv > 0) {
    flux_h_factor = parameters["flux_h_factor"];
  }

  *Fu_avg_->vector() = 0.0;
  *FS_avg_->vector() = 0.0;
  *FT_avg_->vector() = 0.0;
  *Frho_b_avg_->vector() = 0.0;
  *Fmelt_avg_->vector() = 0.0;

  // neccessary for melting part
  Fu_np1_->set_allow_extrapolation(true);
  FS_np1_->set_allow_extrapolation(true);
  FT_np1_->set_allow_extrapolation(true);

  ff_S = std::make_shared<fem_varden_flow::FarFieldTracer>(FS_np1_, *unit_node_normal_,
                                                   parameters);
  ff_T = std::make_shared<fem_varden_flow::FarFieldTracer>(FT_np1_, *unit_node_normal_,
                                                   parameters);

  boundary_layer_thickness =
      static_cast<double>(parameters["boundary_layer_thickness"]);

  friction_vel = std::make_shared<fem_varden_flow::FrictionVelocity>(
      Fu_np1_, *unit_node_normal_, parameters);
  three_eq = std::make_shared<fem_varden_flow::ThreeEquations>(parameters);
  exchangevelocity = std::make_shared<fem_varden_flow::ExchangeVelocity>(parameters);

  dolfin::Array<double> coors_temp(dim_);
  dolfin::Array<double> values_temp(dim_);
  // just choose an arbitrary value inside domain
  for (std::size_t i = 0; i < dim_; i++) {
    coors_temp[i] = 0.0;
    values_temp[i] = 0.0;
  }

  f_T_->eval(values_temp, coors_temp);
  rho_T_factor = values_temp[1];
  f_S_->eval(values_temp, coors_temp);
  rho_S_factor = values_temp[1];
  f_man->eval(values_temp, coors_temp);
  rho_boussinesq_shift = values_temp[1];

  // try to eval Fnormal:
  auto normal_temp = unit_node_normal_->normal();
  normal_temp.eval(values_temp, coors_temp);
  // for some reason this is neccessary to avoid that the program hangs when
  // evaluating far_field

  Fmelt_manifold_ = std::make_shared<Function>(MANIFOLD_EQ_.M);
  Fu_manifold_ = std::make_shared<Function>(MANIFOLD_EQ_.M);

  // Fmelt_manifold_->set_allow_extrapolation(true);
  MANIFOLD_EQ_.ls = std::make_shared<LinearSolver>("cg", "jacobi");
  MANIFOLD_EQ_.ls->parameters["relative_tolerance"] = solver_tolerance_tracer;
  MANIFOLD_EQ_.ls->parameters["absolute_tolerance"] = 1.0e-20;
  MANIFOLD_EQ_.ls->parameters["divergence_limit"] = 1.0e200;
  MANIFOLD_EQ_.ls->parameters["nonzero_initial_guess"] = true;

  // FIXME: is it possible to make this work?
  // MANIFOLD_EQ_.a->ds = sub_domains_;
  // MANIFOLD_EQ_.L->ds = sub_domains_;
  // MANIFOLD_EQ_.a->dx = sub_domains_;
  // MANIFOLD_EQ_.L->dx = sub_domains_;

  auto Czero = std::make_shared<Constant>(0.0);
  auto Cone = std::make_shared<Constant>(1.0);
  MANIFOLD_EQ_.a->set_coefficient("weight", Cmelt_smoothing_weight_);
  if (MANIFOLD_EQ_.A.empty()) {
    assemble(MANIFOLD_EQ_.A, *MANIFOLD_EQ_.a);
  }

  FUNC_->melt_rate->set_coefficient("mb", Cone);
  boundary_surface_ = assemble_functional(*(FUNC_->melt_rate), *scalar);
  std::cout << "boundary surface: " << boundary_surface_ << std::endl;
  FUNC_->melt_rate->set_coefficient("mb", F_melt_);

  std::ostringstream stime;
  stime << "_t=" << time_;
  // file_saver_->add(*F_melting_flux_S_, "melting_flux_S" + stime.str());
  // file_saver_->add(*F_melting_flux_T_, "melting_flux_T" + stime.str());
  file_saver_fine_->add(*F_melt_, M_fine, "melt_fine" + stime.str());
  file_saver_fine_->add(*F_gammaT_, M_fine, "gammaT_fine" + stime.str());
  file_saver_fine_->add(*F_gammaS_, M_fine, "gammaS_fine" + stime.str());
  // file_saver_->add(*Fmelt_manifold_, "melt_manifold" + stime.str());

  // file_saver_fine_->add(*Frho_boussinesq_, M_fine,
  //                      "rho_boussinesq_fine" + stime.str());

  average_parameters = Parameters(parameters);
  average_parameters["t_save"] = average_parameters["t_save_avg"];
  file_saver_fine_average_ =
      std::make_shared<FileSaver>(time_, average_parameters);
  file_saver_fine_average_->add(*Fu_avg_, V_fine, "u_avg_fine");
  file_saver_fine_average_->add(*FT_avg_, M_fine, "T_avg_fine");
  file_saver_fine_average_->add(*FS_avg_, M_fine, "S_avg_fine");
  file_saver_fine_average_->add(*Fmelt_avg_, M_fine, "melt_avg_fine");
  // file_saver_fine_average_->add(*Frho_b_avg_, M_fine, "rho_b_avg_fine");
  file_saver_->add_checkpoint(*FT_avg_, "T_avg");
  file_saver_->add_checkpoint(*FS_avg_, "S_avg");
  file_saver_->add_checkpoint(*Fu_avg_, "u_avg");
  file_saver_->add_checkpoint(*Fmelt_avg_, "melt_avg");
  file_saver_->add_checkpoint(*Frho_b_avg_, "rho_b_avg");
}

//-----------------------------------------------------------------------------
VD_NSESolver::~VD_NSESolver() {
  // Enjoy your tea!
}

//-----------------------------------------------------------------------------
void VD_NSESolver::solve(std::size_t n_retreive) {
  info(parameters, true);
  t_avg_start_ = parameters["t_avg_start"];
  Tend_ = parameters["T"];
  auto Form_kinetic_energy = FUNC_->kinetic_energy;
  double Pavg;
  auto Form_Pavg = FUNC_->Pavg;

  time_ = 0.0;
  dt000_ = dt_;
  dt00_ = dt_;
  dt0_ = dt_;

  auto Fu_analytic = std::make_shared<Function>(MOM_EQ_.V);
  auto Frho_analytic = std::make_shared<Function>(TRACER_EQ_.M);
  auto FP_analytic = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  std::ostringstream stime;

  // Fu_analytic->interpolate(initial_u);
  // Frho_analytic->interpolate(initial_rho_);
  // FP_analytic->interpolate(*initial_P_);

  /// contruct initial data
  if (n_retreive == 0) {
    bool is_checkpoint = false;
    compute_initial(is_checkpoint);
  } else {
    std::vector<double> ch_times;
    file_saver_->get_checkpoint_times(ch_times);

    if (n_retreive == -1) {
      for (int i = 0; i < ch_times.size(); ++i) {
        set_log_active(true);
        info("val[%d] = %g", i, ch_times[i]);
        set_log_active(false);
      }
      return;
    }

    time_ = ch_times[n_retreive];
    file_saver_->read_checkpoint(n_retreive);

    file_saver_->update();
    file_saver_fine_->update();
    file_saver_fine_average_->update();
    bool is_checkpoint = true;
    compute_initial(is_checkpoint);
  }

  dt000_ = dt00_;
  dt00_ = dt0_;
  dt0_ = dt_;

  compute_unknowns();
  Form_kinetic_energy->set_coefficient("u1", Fu_np1_);
  Form_kinetic_energy->set_coefficient("u2", Fu_np1_);
  Form_kinetic_energy->set_coefficient("rho", Frho_np1_);
  Form_kinetic_energy->set_coefficient("T", FT_np1_);
  kinetic_energy = (assemble_functional(*Form_kinetic_energy, *scalar));
  maxDiv = Fdiv_->vector()->max();
  set_log_active(true);
  info("max Div = %.5e, kinetic_energy = %e, l2(p) = %.10e, k = %lf, t = %lf, "
       "iter_time "
       "= %f sec",
       maxDiv, kinetic_energy, 0.0, dt_, time_, toc());
  set_log_active(false);

  std::string file_name = "time_history";
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::ostringstream error_fname;
    std::string strObj = error_fname.str();

    error_fname << file_name.c_str() << ".m";
    std::ofstream errorFile;
    errorFile.open((error_fname.str()).c_str(), std::ios::app);

    errorFile.flush();
    errorFile.close();
  }

  int const time_order = parameters["time_order"];

  if (time_order > 2) {
    constant_dt_init();
  }

  compute_unknowns();
  compute_time_derivatives();

  while (time_ < Tend_ + DOLFIN_EPS_LARGE) {

    time_ += dt_;
    compute_solutions();

    compute_unknowns();
    compute_time_derivatives();

    NV_->compute();

    dt000_ = dt00_;
    dt00_ = dt0_;
    dt0_ = dt_;

    Fu_analytic->interpolate(initial_u);
    Frho_analytic->interpolate(*initial_rho_);
    FP_analytic->interpolate(*initial_P_);

    maxDiv = Fdiv_->vector()->max();

    FUNC_->momentum_->set_coefficient("e_i", e_x_);
    momentum_x = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;

    FUNC_->momentum_->set_coefficient("e_i", e_y_);
    momentum_y = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;

    FUNC_->momentum_->set_coefficient("e_i", e_z_);
    momentum_z = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;

    // / mesh_volume_;
    Form_kinetic_energy->set_coefficient("u1", Fu_np1_);
    Form_kinetic_energy->set_coefficient("u2", Fu_np1_);
    Form_kinetic_energy->set_coefficient("rho", Frho_np1_);
    Form_kinetic_energy->set_coefficient("T", FT_np1_);
    kinetic_energy = (assemble_functional(*Form_kinetic_energy, *scalar));
    angular_momentum_z =
        (assemble_functional(*(FUNC_->angular_momentum_z), *scalar)) -
        functional_offset_;
    mass = (assemble_functional(*(FUNC_->mass), *scalar));

    FUNC_->L2rho->set_coefficient("rho", Frho_np1_);
    double L2rho = (assemble_functional(*(FUNC_->L2rho), *scalar));

    Frho_analytic->vector()->axpy(-1.0, *Frho_np1_->vector());
    FUNC_->L2rho->set_coefficient("rho", Frho_analytic);
    double L2_rho_error = sqrt(assemble_functional(*(FUNC_->L2rho), *scalar));

    if (initial_T_) {
      Frho_analytic->interpolate(*initial_T_);
    }
    Frho_analytic->vector()->axpy(-1.0, *FT_np1_->vector());
    FUNC_->L2rho->set_coefficient("rho", Frho_analytic);
    double L2_T_error = sqrt(assemble_functional(*(FUNC_->L2rho), *scalar));

    Form_kinetic_energy->set_coefficient("u1", Fu_analytic);
    Form_kinetic_energy->set_coefficient("u2", Fu_analytic);
    Form_kinetic_energy->set_coefficient("rho", F_one_);
    Fu_analytic->vector()->axpy(-1.0, *Fu_np1_->vector());

    double L2_u_error =
        sqrt(2. * assemble_functional(*Form_kinetic_energy, *scalar));

    double Linfu = Fu_np1_->vector()->norm("linf");
    double min_rho = Frho_np1_->vector()->min();

    set_log_active(true);
    info("max Div = %.5e, momentum_x = %.5e, angular_momentum_z = %.5e, mass = "
         "%.5e, k = %lf, t = %lf, mom0 iterations = "
         "%i, "
         "rho iterations = %i, min rho = %.5e, kinetic energy = %.5e, "
         "iter_time "
         "= %f sec",
         maxDiv, momentum_x, angular_momentum_z, mass, dt_, time_,
         num_iter_mom0, num_iter_rho, min_rho, kinetic_energy, toc());
    set_log_active(false);

    if (rank == 0) {
      std::ostringstream error_fname;
      std::string strObj = error_fname.str();

      error_fname << file_name.c_str() << ".m";
      std::ofstream errorFile;
      errorFile.open((error_fname.str()).c_str(), std::ios::app);
      // Set the precision to 16 decimals
      errorFile << std::scientific;
      errorFile << std::setprecision(16);

      errorFile << dt_ << "\t" << time_ << "\t" << maxDiv << "\t" << momentum_x
                << "\t" << kinetic_energy << "\t" << Linfu << "\t"
                << num_iter_mom0 << "\t" << L2rho << "\t" << angular_momentum_z
                << "\t" << mass << "\t" << min_rho << "\t" << L2_rho_error
                << "\t" << L2_T_error << "\t" << L2_u_error << "\t" << mass_T
                << "\t" << avg_melt_rate_ << "\t" << tau_ds_ << "\t" << tau_dx_
                << "\n";
      errorFile.flush();
      errorFile.close();
    }

    file_saver_->save();
    file_saver_fine_->save();
    file_saver_->write_checkpoint();
  }
  Form_Pavg->set_coefficient("p", FP_np1_);
  Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
  *FP_np1_->vector() -= Pavg;
}

//-----------------------------------------------------------------------------
void VD_NSESolver::adaptive_solve(std::size_t n_retreive) {
  dolfin::Timer timer_dt("time-step");
  double L2rho, min_rho, L2_rho_error, L2_u_error = 0.0;
  info(parameters, true);
  /// set parameters
  t_avg_start_ = parameters["t_avg_start"];
  Tend_ = parameters["T"];

  double Pavg;
  auto Form_Pavg = FUNC_->Pavg;

  time_ = 0.0;
  dt000_ = dt_;
  dt00_ = dt_;
  dt0_ = dt_;

  auto Fu_analytic = std::make_shared<Function>(MOM_EQ_.V);
  auto Frho_analytic = std::make_shared<Function>(TRACER_EQ_.M);
  auto FP_analytic = std::make_shared<Function>(PRESS_INITIAL_EQ_.Q);
  FP_analytic->interpolate(*initial_P_);
  std::ostringstream stime;

  /// contruct initial data
  if (n_retreive == 0) {
    bool is_checkpoint = false;
    compute_initial(is_checkpoint);
  } else {
    std::vector<double> ch_times;
    file_saver_->get_checkpoint_times(ch_times);

    if (n_retreive == -1) {
      for (int i = 0; i < ch_times.size(); ++i) {
        set_log_active(true);
        info("val[%d] = %g", i, ch_times[i]);
        set_log_active(false);
      }
      return;
    }

    time_ = ch_times[n_retreive];
    file_saver_->read_checkpoint(n_retreive);

    file_saver_->update();
    file_saver_fine_->update();
    file_saver_fine_average_->update();

    bool is_checkpoint = true;
    compute_initial(is_checkpoint);
  }

  compute_unknowns();

  info("l2(u) = %e, l2(p) = %.10e, k = %lf, t = %lf, iter_time = %f sec", 0.0,
       0.0, dt_, time_, toc());

  std::string file_name = "time_history";
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double Linfu = Fu_np1_->vector()->norm("linf");

  if (rank == 0) {
    std::ostringstream error_fname;
    std::string strObj = error_fname.str();

    error_fname << file_name.c_str() << ".m";
    std::ofstream errorFile;
    errorFile.open((error_fname.str()).c_str(), std::ios::app);

    errorFile << dt_ << "\t" << time_ << "\t" << maxDiv << "\t" << momentum_x
              << "\t" << kinetic_energy << "\t" << Linfu << "\t"
              << num_iter_mom0 << "\t" << L2rho << "\t" << angular_momentum_z
              << "\t" << mass << "\t" << min_rho << "\t" << L2_rho_error << "\t"
              << L2_u_error << "\t" << mass_T << "\t" << avg_melt_rate_ << "\t"
              << tau_ds_ << "\t" << tau_dx_ << "\n";
    errorFile.flush();
    errorFile.close();
  }

  double const bdf_init_factor = parameters["bdf_init_factor"];
  dt_ *= bdf_init_factor;
  dt000_ = dt_;
  dt00_ = dt_;
  dt0_ = dt_;

  file_saver_->save();
  file_saver_fine_->save();

  time_ += dt_;
  second_order_iteration();
  compute_unknowns();

  int const time_order = parameters["time_order"];

  if (time_order > 2) {
    time_ += dt_;
    second_order_iteration();
    compute_unknowns();

    NV_->compute_BDF2(*Fu_np1_, *Fu_n_, *Fu_nm1_, dt_, dt0_, *Fu_t_);
    NV_->compute_BDF2(*Frho_np1_, *Frho_n_, *Frho_nm1_, dt_, dt0_, *Frho_t_);
    NV_->compute_BDF2(*FT_np1_, *FT_n_, *FT_nm1_, dt_, dt0_, *FT_t_);
    NV_->compute_BDF2(*FS_np1_, *FS_n_, *FS_nm1_, dt_, dt0_, *FS_t_);
    NV_->compute();

    time_ += dt_;
    third_order_iteration();
    compute_unknowns();

    NV_->compute_BDF2(*Fu_np1_, *Fu_n_, *Fu_nm1_, dt_, dt0_, *Fu_t_);
    NV_->compute_BDF2(*Frho_np1_, *Frho_n_, *Frho_nm1_, dt_, dt0_, *Frho_t_);
    NV_->compute_BDF2(*FT_np1_, *FT_n_, *FT_nm1_, dt_, dt0_, *FT_t_);
    NV_->compute_BDF2(*FS_np1_, *FS_n_, *FS_nm1_, dt_, dt0_, *FS_t_);
    NV_->compute();

    time_ += dt_;
    compute_solutions();
    compute_unknowns();

    NV_->compute_BDF2(*Fu_np1_, *Fu_n_, *Fu_nm1_, dt_, dt0_, *Fu_t_);
    NV_->compute_BDF2(*Frho_np1_, *Frho_n_, *Frho_nm1_, dt_, dt0_, *Frho_t_);
    NV_->compute_BDF2(*FT_np1_, *FT_n_, *FT_nm1_, dt_, dt0_, *FT_t_);
    NV_->compute_BDF2(*FS_np1_, *FS_n_, *FS_nm1_, dt_, dt0_, *FS_t_);
    NV_->compute();
  }

  dt000_ = dt00_;
  dt00_ = dt0_;
  dt0_ = dt_;

  bool redo_step;
  double dt_new = dt_;
  double iter_time = toc();
  while (time_ < Tend_ - 1e-6) {
    timer_dt.start();

    adapt_dt(*Fvloc_, dof_M_, time_, dt_new, parameters, redo_step);

    while (redo_step) {
      time_ -= dt_;
      dt_ = dt_new;
      time_ += dt_;

      redo_compute_solutions();
      compute_unknowns();

      adapt_dt(*Fvloc_, dof_M_, time_, dt_new, parameters, redo_step);

      compute_time_derivatives();
      NV_->compute();
    }

    dt000_ = dt00_;
    dt00_ = dt0_;
    dt0_ = dt_;
    dt_ = dt_new;

    time_ += dt_;
    compute_solutions();

    // compute artificial viscosity
    compute_unknowns();
    compute_time_derivatives();

    // compute RV
    NV_->compute();

    Linfu = Fu_np1_->vector()->norm("linf");
    double min_rho = Frho_np1_->vector()->min();

    iter_time = timer_dt.stop();
    int ETA = std::round((Tend_ - time_) / dt_ * (iter_time));

    set_log_active(true);
    info("u_linf = %.5e, momentum_x = %.5e, angular_momentum_z = %.5e, mass = "
         "%.5e, k = %lf, t = "
         "%lf, mom0 iterations = %i"
         "rho iterations = %i, min_rho = %.5e, iter_time = %f sec, ETA = "
         "%d:%d:%d",
         Linfu, momentum_x, angular_momentum_z, mass, dt_, time_, num_iter_mom0,
         num_iter_rho, min_rho, iter_time, ETA / 3600, (ETA % 3600) / 60,
         ETA % 60);
    set_log_active(false);

    // Move to next time step
    /// save to file
    if (rank == 0) {
      std::ostringstream error_fname;
      std::string strObj = error_fname.str();

      error_fname << file_name.c_str() << ".m";
      std::ofstream errorFile;
      errorFile.open((error_fname.str()).c_str(), std::ios::app);
      errorFile << std::fixed;
      errorFile << std::setprecision(16);
      errorFile << dt_ << "\t" << time_ << "\t" << maxDiv << "\t" << momentum_x
                << "\t" << kinetic_energy << "\t" << Linfu << "\t"
                << num_iter_mom0 << "\t" << L2rho << "\t" << angular_momentum_z
                << "\t" << mass << "\t" << min_rho << "\t" << L2_rho_error
                << "\t" << L2_u_error << "\t" << mass_T << "\t"
                << avg_melt_rate_ << "\t" << tau_ds_ << "\t" << tau_dx_ << "\n";
      errorFile.flush();
      errorFile.close();
    }

    // Fu_analytic->interpolate(initial_u);
    // Frho_analytic->interpolate(*initial_rho_);
    // FP_analytic->interpolate(*initial_P_);

    file_saver_->save();
    file_saver_fine_->save();
    file_saver_->write_checkpoint();
    compute_statistics();
  }

  Form_Pavg->set_coefficient("p", FP_np1_);
  Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
  *FP_np1_->vector() -= Pavg;
}

void VD_NSESolver::compute_initial(bool is_checkpoint) {
  *Frho_np1_->vector() = 1.0;

  if (!is_checkpoint) {
    Fu_np1_->interpolate(initial_u);
    Frho_exact_->interpolate(*initial_rho_);
    Frho_np1_->interpolate(*initial_rho_);

    if (initial_P_ != nullptr) {
      auto Form_Pavg = FUNC_->Pavg;
      double Pavg;
      FP_np1_->interpolate(*initial_P_);
      Form_Pavg->set_coefficient("p", FP_np1_);
      Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
      set_log_active(true);
      // print as many decimals as possible using print():
      info("Pavg = %.20f", Pavg);
      set_log_active(false);
    }

    PDESmooth_rho_ =
        std::make_shared<PDESmooth>(TRACER_EQ_.M, Fh_PX_, parameters);
    PDESmooth_rho_->smooth(Frho_np1_);
    if (!constantTemperature) {
      if (initial_S_ == nullptr) {
        auto zero = std::make_shared<dolfin::Constant>(0.0);
        initial_S_ = zero;
        bcs_S_ = bcs_T_;
      }
      FT_np1_->interpolate(*initial_T_);
      FS_np1_->interpolate(*initial_S_);
      PDESmooth_rho_->smooth(FS_np1_);
      PDESmooth_rho_->smooth(FT_np1_);
      auto Form_Pavg = FUNC_->Pavg;
      Form_Pavg->set_coefficient("p", FT_np1_);
      double Tavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
      Form_Pavg->set_coefficient("p", FP_np1_);
      set_log_active(true);
      // print as many decimals as possible using print():
      info("Tavg = %.20f", Tavg);
      set_log_active(false);
    }
  }

  for (std::size_t i = 0; i < bcs_u_->size(); ++i) {
    (*bcs_u_)[i]->apply(*Fu_np1_->vector());
  }
  *Fu_n_->vector() = *Fu_np1_->vector();
  *Fu_nm1_->vector() = *Fu_np1_->vector();
  *Fu_nm2_->vector() = *Fu_np1_->vector();
  *Fu_nm3_->vector() = *Fu_np1_->vector();

  for (std::size_t i = 0; i < bcs_rho_->size(); ++i) {
    (*bcs_rho_)[i]->apply(*Frho_np1_->vector());
  }
  *Frho_np1_->vector() = *Frho_np1_->vector();
  *Frho_n_->vector() = *Frho_np1_->vector();
  *Frho_nm1_->vector() = *Frho_np1_->vector();
  *Frho_nm2_->vector() = *Frho_np1_->vector();
  *Frho_nm3_->vector() = *Frho_np1_->vector();

  if (!constantTemperature) {
    for (std::size_t i = 0; i < bcs_T_->size(); ++i) {
      (*bcs_T_)[i]->apply(*FT_np1_->vector());
    }
    for (std::size_t i = 0; i < bcs_S_->size(); ++i) {
      (*bcs_S_)[i]->apply(*FS_np1_->vector());
    }
    *FT_n_->vector() = *FT_np1_->vector();
    *FT_nm1_->vector() = *FT_np1_->vector();
    *FT_nm2_->vector() = *FT_np1_->vector();
    *FT_nm3_->vector() = *FT_np1_->vector();
    *FS_np1_->vector() = *FS_np1_->vector();
    *FS_n_->vector() = *FS_np1_->vector();
    *FS_nm1_->vector() = *FS_np1_->vector();
    *FS_nm2_->vector() = *FS_np1_->vector();
    *FS_nm3_->vector() = *FS_np1_->vector();
  }

  // Trying complete boundary as initial pressure BC
  auto zero = std::make_shared<Constant>(0.0);

  std::shared_ptr<Constant> zeros;
  if (dim_ == 2) {
    zeros = std::make_shared<Constant>(0.0, 0.0);
  } else {
    zeros = std::make_shared<Constant>(0.0, 0.0, 0.0);
  }

  compute_UU();
  compute_unknowns();

  get_dt_cfl(*Fvloc_, dof_M_, time_, dt_, parameters);

  dt_ = dt_;

  dt0_ = dt_;
  dt00_ = dt_;
  dt000_ = dt_;

}

void VD_NSESolver::compute_statistics() {
  std::string const demo_type = parameters["demo_type"];

  if (demo_type == "ocean") {
    if (time_ > t_avg_start_ + dt_) {

      *Fu_avg_->vector() *= time_ - t_avg_start_ - dt_;
      *FT_avg_->vector() *= time_ - t_avg_start_ - dt_;
      *FS_avg_->vector() *= time_ - t_avg_start_ - dt_;
      *Fmelt_avg_->vector() *= time_ - t_avg_start_ - dt_;
      *Frho_b_avg_->vector() *= time_ - t_avg_start_ - dt_;

      Fu_avg_->vector()->axpy(dt_, *Fu_np1_->vector());
      FT_avg_->vector()->axpy(dt_, *FT_np1_->vector());
      FS_avg_->vector()->axpy(dt_, *FS_np1_->vector());
      Fmelt_avg_->vector()->axpy(dt_, *F_melt_->vector());
      Frho_b_avg_->vector()->axpy(dt_, *Frho_boussinesq_->vector());

      *Fu_avg_->vector() *= 1 / (time_ - t_avg_start_);
      *FT_avg_->vector() *= 1 / (time_ - t_avg_start_);
      *FS_avg_->vector() *= 1 / (time_ - t_avg_start_);
      *Fmelt_avg_->vector() *= 1 / (time_ - t_avg_start_);
      *Frho_b_avg_->vector() *= 1 / (time_ - t_avg_start_);
    }
    file_saver_fine_average_->save();
  }
}

// -----------------------------------------------------------------------------
void VD_NSESolver::compute_unknowns() {
  tic();
  std::string const demo_type = parameters["demo_type"];

  if (demo_type == "ocean") {

    compute_melting();

    avg_melt_rate_ = (assemble_functional(*(FUNC_->melt_rate), *scalar));

    avg_melt_rate_ /= boundary_surface_;

    double g = parameters["gravitational_acceleration"];
    double rho = parameters["ocean_reference_density"];

    // Hopefully the sign is correct below!
    *Frho_boussinesq_->vector() = rho;
    *Frho_boussinesq_->vector() -= rho_boussinesq_shift / g;
    Frho_boussinesq_->vector()->axpy(-rho * rho_T_factor / g,
                                     *FT_np1_->vector());
    Frho_boussinesq_->vector()->axpy(-rho * rho_S_factor / g,
                                     *FS_np1_->vector());

    // other functionals. Melting functional and so on.
  }

  double maxDivu = Fdiv_->vector()->norm("linf");

  maxDivu = Fdiv_->vector()->max();

  maxDiv = Fdiv_->vector()->max();
  if (demo_type != "ocean") {

    FUNC_->momentum_->set_coefficient("e_i", e_x_);
    momentum_x = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;

    FUNC_->momentum_->set_coefficient("e_i", e_y_);
    momentum_y = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;

    FUNC_->momentum_->set_coefficient("e_i", e_z_);
    momentum_z = (assemble_functional(*(FUNC_->momentum_), *scalar)) -
                 functional_offset_;
    angular_momentum_z =
        (assemble_functional(*(FUNC_->angular_momentum_z), *scalar)) -
        functional_offset_;
  }
  auto Form_kinetic_energy = FUNC_->kinetic_energy;
  Form_kinetic_energy->set_coefficient("u1", Fu_np1_);
  Form_kinetic_energy->set_coefficient("u2", Fu_np1_);
  Form_kinetic_energy->set_coefficient("rho", Frho_np1_);
  Form_kinetic_energy->set_coefficient("T", FT_np1_);
  kinetic_energy = (assemble_functional(*Form_kinetic_energy, *scalar));

  FUNC_->mass->set_coefficient("rho", FT_np1_);
  mass_T = (assemble_functional(*(FUNC_->mass), *scalar));
  *T_bar_ = mass_T / mesh_volume_;

  FUNC_->mass->set_coefficient("rho", FS_np1_);
  mass_S = (assemble_functional(*(FUNC_->mass), *scalar));
  *S_bar_ = mass_S / mesh_volume_;

  FUNC_->mass->set_coefficient("rho", Frho_np1_);
  mass = (assemble_functional(*(FUNC_->mass), *scalar));
  *rho_bar_ = mass / mesh_volume_;

  *C_rho_mass_ = mass / mesh_volume_;
  *C_T_mass_ = mass_T / mesh_volume_;
  *C_S_mass_ = mass_S / mesh_volume_;

  if (remove_rho_average) {

    *Frho_bar_->vector() = mass / mesh_volume_;
    *FT_bar_->vector() = mass_T / mesh_volume_;
    *FS_bar_->vector() = mass_S / mesh_volume_;
  }

  else {
    *Frho_bar_->vector() = 0.0;
    *FT_bar_->vector() = 0.0;
    *FS_bar_->vector() = 0.0;
  }
  set_log_active(true);
  info("Compute unknowns took = %f sec", toc());
  set_log_active(false);
}

//-----------------------------------------------------------------------------
void VD_NSESolver::get_dt_cfl(Function const &Fvloc,
                              std::vector<dolfin::la_index> const &dof_M,
                              double const &t, double &k,
                              Parameters const &parameters) {
  tic();

  int const pol_degree = parameters["PolDegree_V_M"];
  int const cfl_dg0 = parameters["CFL_DG0"];
  double const cfl = parameters["CFL"];
  double const Time = parameters["T"];

  double hvloc_K = 0.;
  double hvloc_max = 0.;
  double h_K = 1000000.;
  double h_min = 1000000.;

  std::vector<double> Fu_arr(dof_V_.size());

  // FIXME: add option to compute timestep based on smoothed velocities
  // std::vector<double> Fu_x_arr(dof_M_.size());
  // std::vector<double> Fu_y_arr(dof_M_.size());
  // std::vector<double> Fu_z_arr(dof_M_.size());

  std::vector<double> Fh_arr(dof_V_.size());
  std::vector<double> Fgrad_arr(dof_V_.size());

  Fu_np1_->vector()->get_local(&Fu_arr[0], dof_V_.size(), &dof_V_[0]);

  // already divided by pol_degree
  Fh_PX_->vector()->get_local(&Fh_arr[0], dof_V_.size(), &dof_V_[0]);

  Fgrad_phi_DG0_->vector()->get_local(&Fgrad_arr[0], dof_DG0_V_.size(),
                                      &dof_DG0_V_[0]);

  // Fvloc.vector()->get_local(&Fvloc_arr[0], dof_M_.size(), &dof_M_[0]);

  std::size_t offset_u = 0;
  std::size_t offset_DG0 = 0;

  std::size_t num_cell_dofs =
      Fu_np1_->function_space()->dofmap()->max_element_dofs();
  std::size_t const value_dim_u = Fu_np1_->value_size();
  std::size_t const num_cell_nodes = num_cell_dofs / value_dim_u;
  std::size_t ii = 0;
  for (CellIterator cell(*Fu_np1_->function_space()->mesh()); !cell.end();
       ++cell) {
    // double h = 1e10;
    // for (EdgeIterator ed(*cell); !ed.end(); ++ed)
    // {
    //   h = std::min(h, ed->length());
    // }
    // h /= static_cast<double>(pol_degree);

    // double vloc_max = 0.0;
    if (cfl_dg0 == 1) {

      for (std::size_t node = 0; node < num_cell_nodes; ++node) {
        // double vloc = 0.0;

        for (int d = 0; d < value_dim_u; ++d) {
          std::size_t uii = offset_u + num_cell_nodes * d + node;
          std::size_t grad_K_pos = offset_DG0 + 1 * d;

          double grad_K_d = Fgrad_arr[grad_K_pos] * pol_degree;
          double u_d = abs(Fu_arr[uii]);
          double h_d = Fh_arr[uii];
          double hvloc_cg = u_d / (h_d + 1.0e-10);
          double hvloc_dg0 = u_d * grad_K_d;
          double hvloc = (1 - cfl_dg0) * hvloc_cg + cfl_dg0 * hvloc_dg0;
          hvloc_max = std::max(hvloc_max, hvloc);
          h_min = std::min(h_min, h_d);
        }
        // vloc = sqrt(vloc);

        // vloc_max = std::max(vloc_max, vloc);

        // double vloc = Fvloc_arr[ii];
        // double h = Fh_arr[ii];
        // double hvloc = vloc / h;
        // h_min = std::min(h_min, h);
      }
    } else {
      Point p = cell->midpoint();
      dolfin::Array<double> coors_midpoint(dim_);
      dolfin::Array<double> u_d(1);
      dolfin::Array<double> h(dim_);
      for (std::size_t d = 0; d < dim_; ++d) {
        coors_midpoint[d] = p[d];
      }

      Fh_PX_->eval(h, coors_midpoint);

      Fvloc_x_->eval(u_d, coors_midpoint);
      hvloc_max = std::max(hvloc_max, u_d[0] / (h[0] + 1.0e-10));
      h_min = std::min(h_min, h[0]);

      Fvloc_y_->eval(u_d, coors_midpoint);
      hvloc_max = std::max(hvloc_max, u_d[0] / (h[1] + 1.0e-10));
      h_min = std::min(h_min, h[1]);

      if (dim_ == 3) {
        Fvloc_z_->eval(u_d, coors_midpoint);
        hvloc_max = std::max(hvloc_max, u_d[0] / (h[2] + 1.0e-10));
        h_min = std::min(h_min, h[2]);
      }
    }

    // hvloc_K = std::max(hvloc_K, vloc_max / h);
    // h_K = std::min(h_K, h);
    offset_u += Fu_np1_->function_space()->dofmap()->max_element_dofs();
    offset_DG0 +=
        Fgrad_phi_DG0_->function_space()->dofmap()->max_element_dofs();
  }

  double k_l = cfl * 1. / (hvloc_max + 1.0e-10);

  k = dolfin::MPI::min(Fu_np1_->function_space()->mesh()->mpi_comm(), k_l);

  double h =
      dolfin::MPI::min(Fu_np1_->function_space()->mesh()->mpi_comm(), h_min);
  if (isinf(k) || k > 1.0e5) {
    k = cfl * h;
  }
}

void VD_NSESolver::adapt_dt(Function const &Fvloc,
                            std::vector<dolfin::la_index> const &dof_M,
                            double const &t, double &k,
                            Parameters const &parameters, bool &redo_step) {

  double const Time = parameters["T"];

  double const kp = parameters["dt_pid_kp"];
  double const ki = parameters["dt_pid_ki"];
  double const kd = parameters["dt_pid_kd"];
  double const safety = parameters["dt_pid_safety"];
  double const err_min = parameters["dt_pid_error_min"];
  double const err_max = parameters["dt_pid_error_max"];
  double ratio_min = parameters["dt_pid_ratio_min"];
  double ratio_max = parameters["dt_pid_ratio_max"];
  double const ratio_retry_min = parameters["dt_pid_retry_ratio_min"];

  ratio_min = std::max(1.0e-6, ratio_min);
  ratio_max = std::max(ratio_min, ratio_max);

  double const k0 = k;

  get_dt_cfl(Fvloc, dof_M, t, k, parameters);
  double const k_cfl = k;

  double s_cfl = (k0 > 0.0) ? (k_cfl / k0) : 1.0;
  if (!std::isfinite(s_cfl) || s_cfl <= 0.0) {
    s_cfl = err_min;
  }

  auto clamp_value = [](double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
  };

  double err_now = (s_cfl > 0.0) ? (1.0 / s_cfl) : err_max;
  err_now = clamp_value(err_now, err_min, err_max);

  double ratio_raw = safety * std::pow(err_now, -kp);

  if (dt_pid_has_prev_error_) {
    double err_prev = clamp_value(dt_pid_prev_error_, err_min, err_max);
    ratio_raw *= std::pow(err_prev, -ki);
    double err_ratio = clamp_value(err_now / err_prev, err_min, err_max);
    ratio_raw *= std::pow(err_ratio, -kd);
  }

  double const ratio_upper_bound = ratio_max;
  double ratio_clamped = clamp_value(ratio_raw, ratio_min, ratio_upper_bound);
  double ratio_final = std::min(ratio_clamped, s_cfl);

  k = k0 * ratio_final;

  redo_step = (ratio_final < ratio_retry_min);

  set_log_active(true);
  info("dt ratio raw = %.3f, clamped = %.3f, s_cfl = %.3f, err = %.3e, redo = "
       "%s",
       ratio_raw, ratio_final, s_cfl, err_now, redo_step ? "true" : "false");
  set_log_active(false);

  if (redo_step) {
  } else {
    dt_pid_prev_error_ = err_now;
    dt_pid_has_prev_error_ = true;

    if (t + k > Time) {
      k = std::fabs(Time - t);
    }
  }
}

void VD_NSESolver::get_rho_min(Function const &Frho,
                               std::vector<dolfin::la_index> const &dof_M,
                               double &rho_min) {
  std::vector<double> Frho_arr(dof_M.size());
  Frho.vector()->get_local(&Frho_arr[0], dof_M.size(), &dof_M[0]);

  std::size_t num_cell_nodes =
      Frho.function_space()->dofmap()->max_element_dofs() / Frho.value_size();
  std::size_t ii = 0;
  for (CellIterator cell(*Frho.function_space()->mesh()); !cell.end(); ++cell) {
    double rho_loc = 99999.9;
    for (std::size_t node = 0; node < num_cell_nodes; ++node, ++ii) {
      rho_loc = std::min(rho_loc, Frho_arr[ii]);
    }

    rho_min = std::min(rho_min, rho_loc);
  }
  rho_min =
      dolfin::MPI::min(Frho.function_space()->mesh()->mpi_comm(), rho_min);
}

std::shared_ptr<Function> &VD_NSESolver::get_Fu() { return Fu_np1_; }

std::shared_ptr<Function> &VD_NSESolver::get_Frho() { return Frho_np1_; }
std::shared_ptr<Function> &VD_NSESolver::get_FT() { return FT_np1_; }
std::shared_ptr<Function> &VD_NSESolver::get_FS() { return FS_np1_; }

std::shared_ptr<double> VD_NSESolver::get_S_bar() { return S_bar_; }
std::shared_ptr<double> VD_NSESolver::get_T_bar() { return T_bar_; }
std::shared_ptr<double> VD_NSESolver::get_rho_bar() { return rho_bar_; }

std::shared_ptr<Function> &VD_NSESolver::get_FP() { return FP_np1_; }
std::shared_ptr<Function> &VD_NSESolver::get_Fh() { return Fh_PX_; }
double VD_NSESolver::get_mesh_volume() { return mesh_volume_; }
double VD_NSESolver::get_final_dt() { return dt_; }

void VD_NSESolver::first_order_iteration() {
  store_previous_solutions();
  time_step_iteration(1);
  return;
}

void VD_NSESolver::second_order_iteration() {
  store_previous_solutions();
  time_step_iteration(2);
  return;
}

void VD_NSESolver::third_order_iteration() {
  store_previous_solutions();
  time_step_iteration(3);
  return;
}

void VD_NSESolver::fourth_order_iteration() {
  store_previous_solutions();
  time_step_iteration(4);
  return;
}

void VD_NSESolver::time_step_iteration(int time_order) {

  bool crank_nicolson = false;
  std::string const demo_type = parameters["demo_type"];

  switch (time_order) {
  case 0:
    break;
  case 1:
    *C_BDF1_ = 1.0;
    *C_crank_nicolson_ = 0.0;
    *C_BDF3_ = 0.0;
    *C_BDF4_ = 0.0;
    error("BDF1 is not implemented yet");
    break;
  case 2:
    *C_BDF1_ = 0.0;
    *C_crank_nicolson_ = 1.0;
    crank_nicolson = true;
    *C_BDF3_ = 0.0;
    *C_BDF4_ = 0.0;
    time_ -= 0.5 * dt_;
    break;
  case 3:
    *C_BDF1_ = 0.0;
    *C_crank_nicolson_ = 0.0;
    *C_BDF3_ = 1.0;
    *C_BDF4_ = 0.0;
    break;
  case 4:
    *C_BDF1_ = 0.0;
    *C_crank_nicolson_ = 0.0;
    *C_BDF3_ = 0.0;
    *C_BDF4_ = 1.0;
    break;
  default:
    break;
  }

  num_iter_rho = 0;

  double Pavg;
  auto Form_Pavg = FUNC_->Pavg;

  *Ck = dt_;
  *Ck0 = dt0_;
  *Ck00 = dt00_;
  *Ck000 = dt000_;

  int newton_iteration_counter = 0;
  int const number_of_newton_iterations =
      parameters["number_of_newton_iterations"];

  *C_max_vms_ = parameters["C_max_vms"];

  num_iter_mom0 = 10;
  while ((num_iter_mom0 > 0 || num_iter_rho > 0) &&
         newton_iteration_counter < number_of_newton_iterations) {

    compute_LPS();

    newton_iteration_counter++;

    if (constant_density == 0) {
      tic();

      TRACER_EQ_.a->set_some_coefficients(coefficients_rho_u);
      TRACER_EQ_.L->set_some_coefficients(coefficients_rho_u);
      assemble(*A_con_, *TRACER_EQ_.a);
      assemble(*b_con_, *TRACER_EQ_.L);

      if (time_order == 2) {
        time_ += 0.5 * dt_;
      }
      // apply Dirichlet bc for density
      for (std::size_t i = 0; i < bcs_rho_->size(); ++i) {
        (*bcs_rho_)[i]->apply(*A_con_, *b_con_);
      }
      if (time_order == 2) {
        time_ -= 0.5 * dt_;
      }

      if (initial_guess_extrapolation == 1 && newton_iteration_counter == 1) {
        NV_->extrapolation4(*Frho_n_, *Frho_nm1_, *Frho_nm2_, *Frho_nm3_, dt_,
                            dt0_, dt00_, dt000_, *Frho_np1_);
      }

      num_iter_rho = lsrho_->solve(*A_con_, *Frho_np1_->vector(), *b_con_);
    }
    if (!constantTemperature) {

      tic();
      if (initial_guess_extrapolation == 1 && newton_iteration_counter == 1) {
        NV_->extrapolation4(*FT_n_, *FT_nm1_, *FT_nm2_, *FT_nm3_, dt_, dt0_,
                            dt00_, dt000_, *FT_np1_);
        NV_->extrapolation4(*FS_n_, *FS_nm1_, *FS_nm2_, *FS_nm3_, dt_, dt0_,
                            dt00_, dt000_, *FS_np1_);

        NV_->extrapolation4(*Fu_n_, *Fu_nm1_, *Fu_nm2_, *Fu_nm3_, dt_, dt0_,
                            dt00_, dt000_, *Fu_np1_);
      }

      set_log_active(true);
      info("Compute unknowns took = %f sec", toc());
      set_log_active(false);
      tic();

      TRACER_EQ_.a->set_some_coefficients(coefficients_only_T);
      TRACER_EQ_.L->set_some_coefficients(coefficients_only_T);

      assemble(*A_con_, *TRACER_EQ_.a);
      assemble(*b_con_, *TRACER_EQ_.L);

      if (time_order == 2) {
        time_ += 0.5 * dt_;
      }
      for (std::size_t i = 0; i < bcs_T_->size(); ++i) {
        (*bcs_T_)[i]->apply(*A_con_, *b_con_);
      }
      if (melt_with_virtual_flux == 0) {
        melt_DBC_T_->apply(*A_con_, *b_con_);
      }
      if (time_order == 2) {
        time_ -= 0.5 * dt_;
      }

      num_iter_rho = lsrho_->solve(*A_con_, *FT_np1_->vector(), *b_con_);

      if (demo_type == "ocean") {

        TRACER_EQ_.a->set_some_coefficients(coefficients_only_S);
        TRACER_EQ_.L->set_some_coefficients(coefficients_only_S);

        assemble(*A_con_, *TRACER_EQ_.a);
        assemble(*b_con_, *TRACER_EQ_.L);

        if (time_order == 2) {
          time_ += 0.5 * dt_;
        }
        for (std::size_t i = 0; i < bcs_S_->size(); ++i) {
          (*bcs_S_)[i]->apply(*A_con_, *b_con_);
        }
        if (melt_with_virtual_flux == 0) {
          melt_DBC_S_->apply(*A_con_, *b_con_);
        }

        if (time_order == 2) {
          time_ -= 0.5 * dt_;
        }

        num_iter_rho = lsrho_->solve(*A_con_, *FS_np1_->vector(), *b_con_);
      }
    }

    set_log_active(true);
    info("Tracer computation took = %f sec", toc());
    set_log_active(false);

    // skip momentum solve for pure advection demo
    if (demo_type != "advection") {

      // FIXME: cleaner way to do this?
      if (demo_type == "ocean") {
        compute_outflow_pressure();
      }

      compute_UU();

      MOM_EQ_.a->set_some_coefficients(
          coefficients_rho_u); // FIXME: remove later
      tic();
      assemble(*A00, *MOM_EQ_.a);

      // Some are already assembled
      assemble(*A10, *a10);
      assemble(*A01, *a01);
      assemble(*A11, *a11);

      assemble(*b0, *MOM_EQ_.L);
      assemble(*b1, *PRESS_INITIAL_EQ_.L);

      double const c_laplace = parameters["C_laplace"];
      double const c_mass = parameters["C_mass"];

      double const w_np3 = dt_ / dt0_;
      double const w_np2 = dt0_ / dt00_;
      double const w_np1 = dt00_ / dt000_;

      double const A1 = 1 + w_np1 * (1 + w_np2);
      double const A2 = 1 + w_np2 * (1 + w_np3);
      double const A3 = 1 + w_np1 * A2;
      double alfa_4 = 1 + w_np3 / (1 + w_np3) + w_np2 * w_np3 / A2 +
                      w_np1 * w_np2 * w_np3 / A3;
      *C_laplace = c_laplace * dt_ / alfa_4;
      *C_mass = c_mass;

      // specific to saddle point
      if (c_brezzi == 0.0) {
        A11->zero();
        // pressure is solved for p^n+1 not p^n+1/2
        b1->zero();
        *C_laplace = c_laplace * dt_ / alfa_4;
        *C_mass = c_mass; //+ 1./nu;
      }

      assemble(*ASchur, *aSchur);

      // apply SlipBC
      if (slip_bcs_) {
        for (std::size_t i = 0; i < slip_bcs_->size(); ++i) {
          (*slip_bcs_)[i]->apply_system(*A00, *A01, *b0);
        }
      }

      if (melt_with_virtual_flux == 0 || melt_with_virtual_flux == 2) {
        inflow_bc_melt_->apply_inflow_system(*A00, *A01, *b0,
                                             *F_melt_VV_->vector());
      }

      if (time_order == 2) {
        time_ += 0.5 * dt_;
      }
      // apply Dirichlet bc
      for (std::size_t i = 0; i < bcs_u_->size(); ++i) {
        (*bcs_u_)[i]->apply(*A00, *b0);
        (*bcs_u_)[i]->zero(*A01);
      }

      // apply Dirichlet bc
      for (std::size_t i = 0; i < bcs_P_->size(); ++i) {
        // (*bcs_P_)[i]->apply(*A11, *b1);
        // (*bcs_P_)[i]->apply(*ASchur, *b1);
        // (*bcs_P_)[i]->zero(*A10);
      }
      if (time_order == 2) {
        time_ -= 0.5 * dt_;
      }

      set_log_active(true);
      info("Assembly of momentum forms took = %f sec", toc());
      set_log_active(false);

      if (initial_guess_extrapolation == 1 && newton_iteration_counter == 1) {
        NV_->extrapolation4(*Fu_n_, *Fu_nm1_, *Fu_nm2_, *Fu_nm3_, dt_, dt0_,
                            dt00_, dt000_, *Fu_np1_);

        NV_->extrapolation3(*FP_n_, *FP_nm1_, *FP_nm2_, dt_, dt0_, dt00_,
                            *FP_np1_);
      }

      tic();
      num_iter_mom0 = schur_krylovsolver->solve(*sol_vec, *rhs_vec);

      Fu_np1_->vector()->apply("insert");
      FP_np1_->vector()->apply("insert");

      if (remove_pressure_average) {
        Form_Pavg->set_coefficient("p", FP_np1_);
        Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
        *FP_np1_->vector() -= Pavg;
      }

      *C_newton_ = 1.0;

      set_log_active(true);

      info("Completed Newton iteration #%i, which took #%i mom and #%i krylov "
           "iterations",
           newton_iteration_counter, num_iter_mom0, num_iter_rho);
      info("Momentum solve took = %f sec", toc());
      set_log_active(false);
    }
  }
  *C_newton_ = 0.0;

}

void VD_NSESolver::constant_dt_init() {

  auto Form_Pavg = FUNC_->Pavg;
  double Pavg;

  time_ += 1 * dt_;

  store_previous_solutions();

  Fu_np1_->interpolate(initial_u);
  Frho_np1_->interpolate(*initial_rho_);
  FP_np1_->interpolate(*initial_P_);

  if (!constantTemperature) {
    FT_np1_->interpolate(*initial_T_);
    FS_np1_->interpolate(*initial_S_);
  }

  if (remove_pressure_average) {
    Form_Pavg->set_coefficient("p", FP_np1_);
    Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
    *FP_np1_->vector() -= Pavg;
  }

  time_ += 1 * dt_;

  store_previous_solutions();

  Fu_np1_->interpolate(initial_u);
  Frho_np1_->interpolate(*initial_rho_);
  FP_np1_->interpolate(*initial_P_);
  if (!constantTemperature) {
    FT_np1_->interpolate(*initial_T_);
    FS_np1_->interpolate(*initial_S_);
  }

  if (remove_pressure_average) {
    Form_Pavg->set_coefficient("p", FP_np1_);
    Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
    *FP_np1_->vector() -= Pavg;
  }

  time_ += 1 * dt_;

  store_previous_solutions();

  Fu_np1_->interpolate(initial_u);
  Frho_np1_->interpolate(*initial_rho_);
  FP_np1_->interpolate(*initial_P_);
  if (!constantTemperature) {
    FT_np1_->interpolate(*initial_T_);
    FS_np1_->interpolate(*initial_S_);
  }

  if (remove_pressure_average) {
    Form_Pavg->set_coefficient("p", FP_np1_);
    Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
    *FP_np1_->vector() -= Pavg;
  }

  time_ += 1 * dt_;
  Fu_np1_->interpolate(initial_u);
  Frho_np1_->interpolate(*initial_rho_);
  FP_np1_->interpolate(*initial_P_);
  if (!constantTemperature) {
    FT_np1_->interpolate(*initial_T_);
    FS_np1_->interpolate(*initial_S_);
  }

  if (remove_pressure_average) {
    Form_Pavg->set_coefficient("p", FP_np1_);
    Pavg = assemble_functional(*Form_Pavg, *scalar) / mesh_volume_;
    *FP_np1_->vector() -= Pavg;
  }

  store_previous_solutions();
}

void VD_NSESolver::redo_compute_solutions() {

  // Store previous solutions
  *Fu_np1_->vector() = *Fu_n_->vector();
  *Frho_np1_->vector() = *Frho_n_->vector();
  *FT_np1_->vector() = *FT_n_->vector();
  *FS_np1_->vector() = *FS_n_->vector();
  *FP_np1_->vector() = *FP_n_->vector();

  compute_solutions();

  // I should actually use previous viscosity here...
}

void VD_NSESolver::compute_solutions() {

  int const time_order = parameters["time_order"];

  switch (time_order) {
  case 1:
    first_order_iteration();
    break;
  case 2:
    second_order_iteration();
    break;
  case 3:
    third_order_iteration();
    break;
  case 4:
    fourth_order_iteration();
    break;
  default:
    error("Time order not implemented");
    break;
  }
}

void VD_NSESolver::compute_time_derivatives() {

  int order = parameters["residual_BDF"];

  switch (order) {
  case 1:
    NV_->compute_BDF1(*Fu_np1_, *Fu_n_, dt_, *Fu_t_);
    NV_->compute_BDF1(*Frho_np1_, *Frho_n_, dt_, *Frho_t_);
    NV_->compute_BDF1(*FT_np1_, *FT_n_, dt_, *FT_t_);
    NV_->compute_BDF1(*FS_np1_, *FS_n_, dt_, *FS_t_);
    break;
  case 2:
    NV_->compute_BDF2(*Fu_np1_, *Fu_n_, *Fu_nm1_, dt_, dt0_, *Fu_t_);
    NV_->compute_BDF2(*Frho_np1_, *Frho_n_, *Frho_nm1_, dt_, dt0_, *Frho_t_);
    NV_->compute_BDF2(*FT_np1_, *FT_n_, *FT_nm1_, dt_, dt0_, *FT_t_);
    NV_->compute_BDF2(*FS_np1_, *FS_n_, *FS_nm1_, dt_, dt0_, *FS_t_);
    break;
  case 3:
    NV_->compute_BDF3(*Fu_np1_, *Fu_n_, *Fu_nm1_, *Fu_nm2_, dt_, dt0_, dt00_,
                      *Fu_t_);
    NV_->compute_BDF3(*Frho_np1_, *Frho_n_, *Frho_nm1_, *Frho_nm2_, dt_, dt0_,
                      dt00_, *Frho_t_);
    NV_->compute_BDF3(*FT_np1_, *FT_n_, *FT_nm1_, *FT_nm2_, dt_, dt0_, dt00_,
                      *FT_t_);
    NV_->compute_BDF3(*FS_np1_, *FS_n_, *FS_nm1_, *FS_nm2_, dt_, dt0_, dt00_,
                      *FS_t_);
    break;
  case 4:
    NV_->compute_BDF4(*Fu_np1_, *Fu_n_, *Fu_nm1_, *Fu_nm2_, *Fu_nm3_, dt_, dt0_,
                      dt00_, dt000_, *Fu_t_);
    NV_->compute_BDF4(*Frho_np1_, *Frho_n_, *Frho_nm1_, *Frho_nm2_, *Frho_nm3_,
                      dt_, dt0_, dt00_, dt000_, *Frho_t_);
    NV_->compute_BDF4(*FT_np1_, *FT_n_, *FT_nm1_, *FT_nm2_, *FT_nm3_, dt_, dt0_,
                      dt00_, dt000_, *FT_t_);
    NV_->compute_BDF4(*FS_np1_, *FS_n_, *FS_nm1_, *FS_nm2_, *FS_nm3_, dt_, dt0_,
                      dt00_, dt000_, *FS_t_);
    break;
  default:
    break;
  }
}

//------------------------------------------------------------------------------
void VD_NSESolver::compute_UU() {

  VLOC_EQ_.L->set_coefficient("u_np1", Fu_np1_);
  VLOC_EQ_.L->set_coefficient("C_directional_wave_speed",
                              C_directional_wave_speed_);
  VLOC_EQ_.L->set_coefficient("C_project_uu", C_zero_);
  VLOC_EQ_.a->set_coefficient("C_project_uu", C_zero_);
  VLOC_EQ_.L->set_coefficient("e_i", e_x_);

  *C_directional_wave_speed_ = 0.0;
  // to make sure velocity is smoothed enough.
  VLOC_EQ_.a->set_coefficient("weight", Crv_smoothing_weight);
  assemble(VLOC_EQ_.b, *VLOC_EQ_.L);
  tic();
  ls_mass_->solve(VLOC_EQ_.A, *Fvloc_->vector(), VLOC_EQ_.b);
  // std::cout << "solve VLOC_EQ_.b" << std::endl;
  Fvloc_->vector()->abs();

  // FIXME: It should be possible to do smoothing and add it to a vector
  // instead. This will be more efficient and look nicer. Need to store a larger
  // matrix though :)
  int directional_wave_speed = parameters["directional_wave_speed"];
  if (directional_wave_speed == 1) {
    *C_directional_wave_speed_ = 1.0;
    VLOC_EQ_.L->set_coefficient("e_i", e_x_);
    assemble(VLOC_EQ_.b, *VLOC_EQ_.L);
    tic();
    ls_mass_->solve(VLOC_EQ_.A, *Fvloc_x_->vector(), VLOC_EQ_.b);
    // Fvloc_x_->vector()->abs();

    VLOC_EQ_.L->set_coefficient("e_i", e_y_);
    assemble(VLOC_EQ_.b, *VLOC_EQ_.L);
    tic();
    ls_mass_->solve(VLOC_EQ_.A, *Fvloc_y_->vector(), VLOC_EQ_.b);
    // Fvloc_y_->vector()->abs();

    if (dim_ == 3) {
      VLOC_EQ_.L->set_coefficient("e_i", e_z_);
      assemble(VLOC_EQ_.b, *VLOC_EQ_.L);
      ls_mass_->solve(VLOC_EQ_.A, *Fvloc_z_->vector(), VLOC_EQ_.b);
      // Fvloc_z_->vector()->abs();
    }
  } else {
    *Fvloc_x_->vector() = *Fvloc_->vector();
    *Fvloc_y_->vector() = *Fvloc_->vector();
    *Fvloc_z_->vector() = *Fvloc_->vector();
  }

  if (project_uu) {
    VLOC_EQ_.a->set_coefficient("C_project_uu", C_one_);
    VLOC_EQ_.L->set_coefficient("C_project_uu", C_one_);
    assemble(VLOC_EQ_.A, *VLOC_EQ_.a);
    assemble(VLOC_EQ_.b, *VLOC_EQ_.L);
    ls_mass_->solve(VLOC_EQ_.A, *F_uu_projected_->vector(), VLOC_EQ_.b);

    VLOC_EQ_.a->set_coefficient("C_project_uu", C_zero_);
    VLOC_EQ_.L->set_coefficient("C_project_uu", C_zero_);
    assemble(VLOC_EQ_.A, *VLOC_EQ_.a);
  }
}

// FIXME: it would be faster to solve outflow pressure on the manifold. But it
// is not so slow this way so it is OK
void VD_NSESolver::compute_outflow_pressure() {  
  tic();

  OUTFLOW_EQ_.L->set_some_coefficients(coefficients_rho_u);
  OUTFLOW_EQ_.L->set_some_coefficients(coefficients_T);
  OUTFLOW_EQ_.L->set_some_coefficients(coefficients_S);

  if (OUTFLOW_EQ_.A.empty()) {
    assemble(OUTFLOW_EQ_.A, *OUTFLOW_EQ_.a);
  }

  assemble(OUTFLOW_EQ_.b, *OUTFLOW_EQ_.L);

  // apply Dirichlet bc
  for (std::size_t i = 0; i < bcs_P_->size(); ++i) {
    // FIXME: now i overrided the usual pressure bc since i am not using that.
    // There is no way to put Dirichlet BC on pressure at outflow right now...
    // :/
    (*bcs_P_)[i]->apply(OUTFLOW_EQ_.A, OUTFLOW_EQ_.b);
  }

  if (OUTFLOW_EQ_.ls == nullptr) {
    double solver_tolerance_tracer = parameters["solver_tolerance_tracer"];
    OUTFLOW_EQ_.ls = std::make_shared<LinearSolver>("cg", "hypre_amg");
    OUTFLOW_EQ_.ls->parameters["relative_tolerance"] = solver_tolerance_tracer;
    OUTFLOW_EQ_.ls->parameters["absolute_tolerance"] = 1.0e-20;
    OUTFLOW_EQ_.ls->parameters["nonzero_initial_guess"] = true;
  }
  int iteration_outflow = OUTFLOW_EQ_.ls->solve(
      OUTFLOW_EQ_.A, *FP_outflow_->vector(), OUTFLOW_EQ_.b);
  set_log_active(true);
  info("Outflow pressure solve took %i iterations and %f sec.",
       iteration_outflow, toc());
  set_log_active(false);
}

void VD_NSESolver::compute_LPS() {
  if (*C_max_vms_ == 0.0) {
    return;
  }
  tic();
  std::string const demo_type = parameters["demo_type"];

  LPS_EQ_.L->set_some_coefficients(coefficients_rho_u);
  if (constant_density == 0) {
    assemble(LPS_EQ_.b, *LPS_EQ_.L);
    LPS_EQ_.ls->solve(LPS_EQ_.A, *Fgrad_rho_->vector(), LPS_EQ_.b);
  }

  if (demo_type == "advection") {
    return;
  }

  if (!constantTemperature) {

    LPS_EQ_.L->set_some_coefficients(coefficients_only_T);
    assemble(LPS_EQ_.b, *LPS_EQ_.L);
    LPS_EQ_.ls->solve(LPS_EQ_.A, *Fgrad_T_->vector(), LPS_EQ_.b);

    LPS_EQ_.L->set_some_coefficients(coefficients_only_S);
    assemble(LPS_EQ_.b, *LPS_EQ_.L);
    LPS_EQ_.ls->solve(LPS_EQ_.A, *Fgrad_S_->vector(), LPS_EQ_.b);
  }

  
  if (c_brezzi > 0) {
    LPS_EQ_.L->set_some_coefficients(coefficients_only_P);
    assemble(LPS_EQ_.b, *LPS_EQ_.L);
    LPS_EQ_.ls->solve(LPS_EQ_.A, *Fgrad_p_->vector(), LPS_EQ_.b);
  }

  LPS_TENSOR_EQ_.L->set_some_coefficients(coefficients_rho_u);
  assemble(LPS_TENSOR_EQ_.b, *LPS_TENSOR_EQ_.L);
  LPS_TENSOR_EQ_.ls->solve(LPS_TENSOR_EQ_.A, *Fgrad_u_->vector(),
                           LPS_TENSOR_EQ_.b);
  set_log_active(true);
  info("LPS solve took = %f sec", toc());
  set_log_active(false);
}

void VD_NSESolver::compute_melting() {

  int rank;
  MPI_Comm_rank(mesh_->mpi_comm(), &rank);

  dolfin::Array<double> coors(dim_);
  dolfin::Array<double> tracer_value_T(1);
  dolfin::Array<double> tracer_value_S(1);
  dolfin::Array<double> ff_S_value(1);
  dolfin::Array<double> ff_T_value(1);
  dolfin::Array<double> friction_value(1);
  dolfin::Array<double> vloc_value(1);
  dolfin::Array<double> u_values(dim_);
  dolfin::Array<double> u_value(1);
  dolfin::Array<double> p_value(1);
  dolfin::Array<double> h_value(dim_);
  dolfin::Array<double> res_S_value(1);
  dolfin::Array<double> res_T_value(1);
  dolfin::Array<double> res_u_value(1);

  dolfin::Array<double> ff_S_values(dim_);   // needed for tensor valued rv
  dolfin::Array<double> ff_T_values(dim_);   // needed for tensor valued rv
  dolfin::Array<double> normal_values(dim_); // needed for tensor valued rv
  dolfin::Array<double> kappa_T_h(dim_);     // needed for tensor valued rv
  dolfin::Array<double> kappa_S_h(dim_);     // needed for tensor valued rv
  std::vector<double> solution;

  double c_max = (double)*C_max_;
  double kappa_T = (double)*C_kappa_T_;
  double kappa_S = (double)*C_kappa_S_;
  double nu = (double)*Cnu_;

  double sc_T = nu / kappa_T;
  double sc_S = nu / kappa_S;

  auto fe = M_->element();
  auto dm = M_->dofmap();
  // Owned range of dofs on this rank
  auto range = F_melt_->vector()->local_range();
  std::size_t start = range.first, end = range.second;
  std::size_t n_owned = end - start;

  auto melt_arr = std::vector<double>();
  auto s_b_arr = std::vector<double>();
  auto t_b_arr = std::vector<double>();
  auto gammaT_arr = std::vector<double>();
  auto gammaS_arr = std::vector<double>();
  auto flux_arr_S = std::vector<double>();
  auto flux_arr_T = std::vector<double>();
  melt_arr.resize(n_owned, 0.0);   // resize to the owned range size
  s_b_arr.resize(n_owned, 0.0);    // resize to the owned range size
  t_b_arr.resize(n_owned, 0.0);    // resize to the owned range size
  flux_arr_S.resize(n_owned, 0.0); // resize to the owned range size
  flux_arr_T.resize(n_owned, 0.0); // resize to the owned range size

  F_melt_->vector()->get_local(melt_arr);
  F_gammaT_->vector()->get_local(gammaT_arr);
  F_gammaS_->vector()->get_local(gammaS_arr);
  F_melting_flux_S_->vector()->get_local(flux_arr_S);
  F_melting_flux_T_->vector()->get_local(flux_arr_T);

  for (CellIterator c(*mesh_); !c.end(); ++c) {
    auto cell_facets =
        c->entities(dim_ - 1); // global facet indices for this cell
    std::size_t nfacets = c->num_entities(dim_ - 1); // edges in 2D, faces in 3D
    auto gdofs = dm->cell_dofs(c->index());
    std::vector<double> dof_coords;
    dof_coords = M_->tabulate_dof_coordinates();

    double u_prev = 0.0;
    std::vector<double> s_m_prev_values(dim_, mass_S / mesh_volume_);
    std::vector<double> t_m_prev_values(dim_, mass_T / mesh_volume_);
    for (std::size_t lf = 0; lf < nfacets; ++lf) {
      std::size_t global_facet_index = cell_facets[lf];
      Facet facet(*mesh_, global_facet_index);

      // skip facets not on boundary and not on ice-shelf
      if (!facet.exterior() || (*sub_domains_)[global_facet_index] != 1) {
        continue;
      }
      std::vector<std::size_t> ldofs;
      dm->tabulate_facet_dofs(ldofs, lf);

      for (auto ld : ldofs) {
        std::size_t dof = gdofs[ld];

        // Only write if this rank OWNS the dof
        if (dof < 0 || dof >= end - start)
          continue;

        // Coordinates of this dof:
        for (std::size_t i = 0; i < dim_; ++i) {
          coors[i] = dof_coords[dim_ * dof + i];
        }

        // initialize all as zero
        double gamma_S{}, gamma_T{}, flux_T{}, flux_S{}, mb{};

        // FIXME: compute pressure in a better way! right now we assume that the
        // atm coordinate has to be a certain way... g*rho*water_depth
        double rho_avg = parameters["water_density"]; // 1027.6;
        // double rho = 999.8;
        double g = 9.81;
        // Fvloc_->eval(vloc_value, coors);
        // FP_->eval(p_value, coors);
        // double p_numerical = p_value[0];
        // double vloc = vloc_value[0];
        // double y = (1000.0 - coors[1]);
        // double p = rho*( p_numerical - g*y + 0.5*vloc -
        // 0.5*y*(rho_T_factor*Tm + rho_S_factor*Sm) );

        // numerical P and simple p gives almost the same value
        double P = g * rho_avg * std::abs(coors[1]);

        if (melting_with_rv > 0) {
          Fh_PX_->eval(h_value, coors);

          double h = 0.0;
          for (std::size_t i = 0; i < dim_; i++) {
            h += h_value[i] * h_value[i];
          }
          h = flux_h_factor * sqrt(h);

          ff_S->update_bl_thickness(h);
          ff_T->update_bl_thickness(h);

          // using Stefan + RV model
          Fvloc_x_->eval(u_value, coors);
          u_values[0] = u_value[0];
          Fvloc_y_->eval(u_value, coors);
          u_values[1] = u_value[0];
          if (dim_ == 3) {
            Fvloc_z_->eval(u_value, coors);
            u_values[2] = u_value[0];
          }

          NV_->get_res_S()->eval(res_S_value, coors);
          NV_->get_res_T()->eval(res_T_value, coors);
          NV_->get_res_u()->eval(res_u_value, coors);

          Fnormal_->eval(normal_values, coors);
          double u_n = 0.0;
          if (melting_with_rv == 1) {

            for (int i = 0; i < dim_; i++) {
              u_n = u_values[i] * normal_values[i];
            }
          } else if (melting_with_rv == 2) {
            // scalar valued RV
            double vloc = 0.0;
            for (std::size_t i = 0; i < dim_; i++) {
              vloc += u_values[i] * u_values[i];
            }
            vloc = sqrt(vloc);
            u_n = vloc;
          }

          gamma_S = (kappa_S / h + c_max * abs(u_n) * res_u_value[0] * 1 /
                                       (std::pow(sc_S, 2.0 / 3.0)));
          gamma_T = (kappa_T / h + c_max * abs(u_n) * res_u_value[0] * 1 /
                                       (std::pow(sc_T, 2.0 / 3.0)));

        } else {
          friction_vel->eval(friction_value, coors);

          if (isnan(friction_value[0])) {
            set_log_active(true);
            dolfin::warning("friction velocity at index %d is nan. Setting to "
                            "previous value u_prev = %f",
                            dof, u_prev);
            set_log_active(false);
            friction_value[0] = u_prev;
          }
          u_prev = friction_value[0];

          // using 3-eq formulation
          gamma_S = exchangevelocity->gammaS(friction_value[0]);
          gamma_T = exchangevelocity->gammaT(friction_value[0]);
        }

        s_m_prev_values[0] = ff_S_values[0];
        t_m_prev_values[0] = ff_T_values[0];
        ff_S->eval(ff_S_value, coors);
        ff_T->eval(ff_T_value, coors);
        double Sm = ff_S_value[0];
        double Tm = ff_T_value[0];

        if (isnan(Sm)) {
          Sm = s_m_prev_values[0];
          set_log_active(true);
          dolfin::warning(
              "Sm at index %d is nan. Setting Sm to previous value = %f", dof,
              Sm);
          set_log_active(false);
        }

        if (isnan(Tm)) {
          Tm = t_m_prev_values[0];
          set_log_active(true);
          dolfin::warning(
              "Tm at index %d is nan. Setting Tm to previous value = %f", dof,
              Tm);
          set_log_active(false);
        }

        solution = three_eq->solve(Sm, Tm, P, gamma_S, gamma_T, false);
        double Sb = solution[0];
        double Tb = solution[1];
        mb = solution[2];

        if (melt_with_virtual_flux == 1) {
          flux_S = (gamma_S + mb) * (Sm - Sb);
          flux_T = (gamma_T + mb) * (Tm - Tb);
        } else if (melt_with_virtual_flux == 2) {
          // volume flux according to Assay-Davis et al 2016
          flux_S = 0.0; // Lukas: It feels more reasonable if it would be:
                        // gamma_S * (Sm - Sb) - mb * Sb;
          flux_T = gamma_T * (Tm - Tb) - mb * Tb;
        }
        // no flux with melt_with_virtual_flux == 0

        melt_arr[dof] = mb;
        gammaT_arr[dof] = gamma_T;
        gammaS_arr[dof] = gamma_S;
        flux_arr_S[dof] = flux_S;
        flux_arr_T[dof] = flux_T;
        s_b_arr[dof] = Sb;
        t_b_arr[dof] = Tb;
      }
    }
  }

  F_melt_->vector()->set_local(melt_arr);
  F_melt_->vector()->apply("insert");
  F_gammaT_->vector()->set_local(gammaT_arr);
  F_gammaT_->vector()->apply("insert");
  F_gammaS_->vector()->set_local(gammaS_arr);
  F_gammaS_->vector()->apply("insert");

  F_melting_flux_S_->vector()->set_local(flux_arr_S);
  F_melting_flux_S_->vector()->apply("insert");
  F_melting_flux_T_->vector()->set_local(flux_arr_T);
  F_melting_flux_T_->vector()->apply("insert");

  const int smooth_melting = parameters["smooth_melting"];

  if (smooth_melting >= 1) {
    MANIFOLD_EQ_.L->set_coefficient("m_", Fmelt_manifold_);

    Fmelt_manifold_->interpolate(*F_melt_);
    assemble(MANIFOLD_EQ_.b, *MANIFOLD_EQ_.L);
    MANIFOLD_EQ_.ls->solve(MANIFOLD_EQ_.A, *Fmelt_manifold_->vector(),
                           MANIFOLD_EQ_.b);
    interpolate_from_manifold_to_domain(F_melt_, Fmelt_manifold_);

    Fmelt_manifold_->interpolate(*F_melting_flux_S_);
    assemble(MANIFOLD_EQ_.b, *MANIFOLD_EQ_.L);
    MANIFOLD_EQ_.ls->solve(MANIFOLD_EQ_.A, *Fmelt_manifold_->vector(),
                           MANIFOLD_EQ_.b);
    interpolate_from_manifold_to_domain(F_melting_flux_S_, Fmelt_manifold_);

    Fmelt_manifold_->interpolate(*F_melting_flux_T_);
    assemble(MANIFOLD_EQ_.b, *MANIFOLD_EQ_.L);
    MANIFOLD_EQ_.ls->solve(MANIFOLD_EQ_.A, *Fmelt_manifold_->vector(),
                           MANIFOLD_EQ_.b);
    interpolate_from_manifold_to_domain(F_melting_flux_T_, Fmelt_manifold_);

    // Do interpolation manually because LagrangeInterpolation is slow
  }

  if (melt_with_virtual_flux == 0) {

    FS_b_->vector()->set_local(s_b_arr);
    FS_b_->vector()->apply("insert");
    FT_b_->vector()->set_local(t_b_arr);
    FT_b_->vector()->apply("insert");

    // *F_melt_->vector() = 1.0e-3; // works !
    F_melt_V_->interpolate(*F_melt_);
    melt_vector_bc->apply(*F_melt_V_->vector());

    melt_vector_bc0->apply(*F_melt_VV_->vector());
    melt_vector_bc1->apply(*F_melt_VV_->vector());

    if (dim_ == 3) {
      melt_vector_bc2->apply(*F_melt_VV_->vector());
    }
  }
}

void VD_NSESolver::interpolate_from_manifold_to_domain(
    std::shared_ptr<Function> F_domain, std::shared_ptr<Function> F_manifold) {

  auto fe = M_->element();
  auto dm = M_->dofmap();
  // Owned range of dofs on this rank
  auto range = F_domain->vector()->local_range();
  std::size_t start = range.first, end = range.second;
  std::size_t n_owned = end - start;

  auto domain_arr = std::vector<double>();
  domain_arr.resize(n_owned, 0.0); // resize to the owned range size

  F_domain->vector()->get_local(domain_arr);

  for (CellIterator c(*mesh_); !c.end(); ++c) {
    // tabulate_entity_dofs();
    auto cell_facets = c->entities(1); // global facet indices for this cell
    std::size_t nfacets = c->num_entities(1);
    auto gdofs = dm->cell_dofs(c->index());
    std::vector<double> dof_coords;
    dolfin::Array<double> manifold_value(1);
    dolfin::Array<double> coors(dim_);
    dof_coords = M_->tabulate_dof_coordinates();

    for (std::size_t lf = 0; lf < nfacets; ++lf) {
      std::size_t global_facet_index = cell_facets[lf];
      Facet facet(*mesh_, global_facet_index);

      // skip facets not on boundary and not on ice-shelf
      if (!facet.exterior() || (*sub_domains_)[global_facet_index] != 1) {
        continue;
      }

      std::vector<std::size_t> ldofs;
      dm->tabulate_facet_dofs(ldofs, lf);

      for (auto ld : ldofs) {
        std::size_t dof = gdofs[ld];

        // Only write if this rank OWNS the dof
        if (dof < 0 || dof >= end - start)
          continue;

        // Coordinates of this dof:
        for (std::size_t i = 0; i < dim_; ++i) {
          coors[i] = dof_coords[dim_ * dof + i];
        }

        F_manifold->eval(manifold_value, coors);

        domain_arr[dof] = manifold_value[0];
      }
    }
  }

  F_domain->vector()->set_local(domain_arr);
  F_domain->vector()->apply("insert");
}

void VD_NSESolver::store_previous_solutions() {

  *Fu_nm3_->vector() = *Fu_nm2_->vector();
  *Fu_nm2_->vector() = *Fu_nm1_->vector();
  *Fu_nm1_->vector() = *Fu_n_->vector();
  *Fu_n_->vector() = *Fu_np1_->vector();

  *FP_nm3_->vector() = *FP_nm2_->vector();
  *FP_nm2_->vector() = *FP_nm1_->vector();
  *FP_nm1_->vector() = *FP_n_->vector();
  *FP_n_->vector() = *FP_np1_->vector();

  *Frho_nm3_->vector() = *Frho_nm2_->vector();
  *Frho_nm2_->vector() = *Frho_nm1_->vector();
  *Frho_nm1_->vector() = *Frho_n_->vector();
  *Frho_n_->vector() = *Frho_np1_->vector();

  *FT_nm3_->vector() = *FT_nm2_->vector();
  *FT_nm2_->vector() = *FT_nm1_->vector();
  *FT_nm1_->vector() = *FT_n_->vector();
  *FT_n_->vector() = *FT_np1_->vector();

  *FS_nm3_->vector() = *FS_nm2_->vector();
  *FS_nm2_->vector() = *FS_nm1_->vector();
  *FS_nm1_->vector() = *FS_n_->vector();
  *FS_n_->vector() = *FS_np1_->vector();
}

// not used
void VD_NSESolver::relaxation_conservation_fix() {

  int const time_order = parameters["time_order"];
  auto Form_kinetic_energy = FUNC_->kinetic_energy;
  double time0 = time_ - dt_;

  // relaxation to ensure conservation of energy even with bad tolerance
  // only for second order scheme
  if (time_order != 2)
    return;

  auto Fu_temp = std::make_shared<Function>(MOM_EQ_.V);
  *Fu_temp->vector() = 0.0;

  Form_kinetic_energy->set_coefficient("u1", Fu_n_);
  Form_kinetic_energy->set_coefficient("u2", Fu_n_);
  double num1 = 2.0 * (assemble_functional(*Form_kinetic_energy, *scalar));

  Form_kinetic_energy->set_coefficient("u1", Fu_np1_);
  Form_kinetic_energy->set_coefficient("u2", Fu_n_);
  double num2 = 2.0 * (assemble_functional(*Form_kinetic_energy, *scalar));

  Fu_temp->vector()->axpy(1.0, *Fu_np1_->vector());
  Fu_temp->vector()->axpy(-1.0, *Fu_n_->vector());
  Form_kinetic_energy->set_coefficient("u1", Fu_temp);
  Form_kinetic_energy->set_coefficient("u2", Fu_temp);
  double den = 2.0 * (assemble_functional(*Form_kinetic_energy, *scalar));

  double relax_factor = 2.0 * (num1 - num2) / den;

  set_log_active(true);
  info("Relaxation factor = %f", relax_factor);
  set_log_active(false);
  *Fu_temp->vector() = 0.0;
  Fu_temp->vector()->axpy(relax_factor, *Fu_np1_->vector());
  Fu_temp->vector()->axpy(-relax_factor, *Fu_n_->vector());
  Fu_temp->vector()->axpy(1.0, *Fu_n_->vector());
  time_ = time0 + relax_factor * (time_ - time0);

  *Fu_np1_->vector() = 0.0;
  Fu_np1_->vector()->axpy(1.0, *Fu_temp->vector());
}

double VD_NSESolver::assemble_functional(const Form &a, Scalar &s) {
  if (a.rank() != 0) {
    dolfin_error("assemble.cpp", "assemble form",
                 "Expecting a scalar form but rank is %d", a.rank());
  }
  // Create Scalar with appropriate communicator
  dolfin_assert(a.mesh());
  // Assemble
  Assembler assembler;
  assembler.assemble(s, a);
  return s.get_scalar_value();
}

// Mark bad cells based on some criteria. Not used currently.
void VD_NSESolver::add_noslip_on_bad_cells() {

  double threshold_bad_cell = parameters["threshold_bad_cell"];
  std::shared_ptr<Form> boundary_marker_a;
  std::shared_ptr<Form> boundary_marker_L;
  std::shared_ptr<FunctionSpace> V_temp;
  if (dim_ == 2) {
    V_temp = std::make_shared<BoundaryMarker2D::FunctionSpace>(mesh_);
    boundary_marker_a =
        std::make_shared<BoundaryMarker2D::BilinearForm>(V_temp, V_temp);
    boundary_marker_L = std::make_shared<BoundaryMarker2D::LinearForm>(V_temp);
  } else if (dim_ == 3) {
    V_temp = std::make_shared<BoundaryMarker3D::FunctionSpace>(mesh_);
    boundary_marker_a =
        std::make_shared<BoundaryMarker3D::BilinearForm>(V_temp, V_temp);
    boundary_marker_L = std::make_shared<BoundaryMarker3D::LinearForm>(V_temp);
  }

  dolfin::Matrix A;
  dolfin::Vector b;

  const int bad_cells_boundary_condition =
      parameters["bad_cells_boundary_condition"];
  auto C_directional = std::make_shared<Constant>(0.0);

  if (bad_cells_boundary_condition == 2) {
    *C_directional = 1.0;
  }

  boundary_marker_a->set_coefficient("h", Fh_PX_);
  boundary_marker_a->set_coefficient("weight", Crv_smoothing_weight);
  boundary_marker_a->set_coefficient("C_directional", C_directional);
  assemble(A, *boundary_marker_a);
  assemble(b, *boundary_marker_L);
  auto F_bad_cells_cg1_ = std::make_shared<Function>(V_temp);

  auto ls_mass_temp = std::make_shared<LinearSolver>("cg", "jacobi");
  ls_mass_temp->parameters["relative_tolerance"] = 1.0e-12;
  ls_mass_temp->parameters["absolute_tolerance"] = 1.0e-20;
  ls_mass_temp->parameters["nonzero_initial_guess"] = true;
  ls_mass_temp->solve(A, *F_bad_cells_cg1_->vector(), b);

  dolfin::Array<double> value_array(1);
  const std::size_t gdim = mesh_->geometry().dim();
  dolfin::Array<double> point_array(gdim);
  for (dolfin::CellIterator cell(*mesh_); !cell.end(); ++cell) {
    const auto midpoint = cell->midpoint();

    for (std::size_t i = 0; i < gdim; ++i) {
      point_array[i] = midpoint[i];
    }
    F_bad_cells_cg1_->eval(value_array, point_array);
    if (value_array[0] > threshold_bad_cell) {
      (*marker_bad_cells_)[cell->index()] = 1;
    } else {
      (*marker_bad_cells_)[cell->index()] = 0;
    }
  }

  File file_bad_cells("results/pvd_files/bad_cells.pvd");
  file_bad_cells.write(*marker_bad_cells_);
}
