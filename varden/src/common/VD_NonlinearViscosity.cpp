// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <dolfin/common/timing.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/io/File.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/refinement/refine.h>

#include <utility>

#include "DofMappings.h"

#include "VD_NSEResidual.h"
#include "VD_NonlinearViscosity.h"

///
using dolfin::EdgeIterator;
using dolfin::LinearSolver;
using dolfin::MPI;

using namespace fem_varden_flow;

//-----------------------------------------------------------------------------
VD_NonlinearViscosity::VD_NonlinearViscosity(
    std::shared_ptr<const Mesh> mesh, std::vector<dolfin::la_index> &dof_M,
    std::vector<dolfin::la_index> &dof_V, Parameters &parameters)
    : parameters(parameters), dof_M_(dof_M), dof_V_(dof_V), mesh_(std::move(mesh)) {
  configure_residual(std::shared_ptr<const FunctionSpace>());
  init();
}

//-----------------------------------------------------------------------------
VD_NonlinearViscosity::VD_NonlinearViscosity(
    std::shared_ptr<const Mesh> mesh, std::shared_ptr<const FunctionSpace> M,
    std::vector<dolfin::la_index> &dof_M, std::vector<dolfin::la_index> &dof_V,
    Parameters &parameters)
    : parameters(parameters), dof_M_(dof_M), dof_V_(dof_V), mesh_(std::move(mesh)) {
  configure_residual(M);
  init();
}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::configure_residual(
    const std::shared_ptr<const FunctionSpace>& M) {

  auto init_residual = [&](const std::shared_ptr<const FunctionSpace>& space) {
    if (space) {
      VD_NSEResidual::init(mesh_, space, RES, parameters);
    } else {
      VD_NSEResidual::init(mesh_, RES, parameters);
    }
  };


    init_residual(M);
    ls_ = std::make_shared<LinearSolver>("cg", "jacobi");
    double const tolerance = parameters["solver_tolerance_tracer"];
    ls_->parameters["relative_tolerance"] = tolerance;
    ls_->parameters["absolute_tolerance"] = 1.0e-20;
    ls_->parameters["nonzero_initial_guess"] = true;
    ls_->parameters["divergence_limit"] = 1.0e15;
    return;

}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::init() {

  Fmom_loc = std::make_shared<Function>(RES.M);
  FT_u_loc = std::make_shared<Function>(RES.M);
  FS_u_loc = std::make_shared<Function>(RES.M);
  Fkinetic_loc = std::make_shared<Function>(RES.M);
  Fres_u_ = std::make_shared<Function>(RES.M);
  Fres_rho_ = std::make_shared<Function>(RES.M);
  Fres_T_ = std::make_shared<Function>(RES.M);
  Fres_S_ = std::make_shared<Function>(RES.M);


  Norm_u = std::make_shared<Constant>(1.0);
  Norm_mom = std::make_shared<Constant>(1.0);
  Norm_T_u = std::make_shared<Constant>(1.0);
  Norm_S_u = std::make_shared<Constant>(1.0);
  Norm_rho = std::make_shared<Constant>(1.0);
  Norm_kinetic = std::make_shared<Constant>(1.0);

  Cthreshold = std::make_shared<Constant>(parameters["C_flat"]);
  Cgamma = std::make_shared<Constant>(parameters["rv_gamma"]);
  C_R = std::make_shared<Constant>(parameters["C_R"]);
  C_res_rho = std::make_shared<Constant>(1.0);
  C_res_u = std::make_shared<Constant>(1.0);
  C_res_T = std::make_shared<Constant>(1.0);

  RES.L->set_coefficient("Norm_u_inf", Norm_u);
  RES.L->set_coefficient("Norm_mom_inf", Norm_mom);
  RES.L->set_coefficient("Norm_k_inf", Norm_kinetic);
  RES.L->set_coefficient("threshold", Cthreshold);
  RES.L->set_coefficient("gamma", Cgamma);
  RES.L->set_coefficient("C_R", C_R);
  RES.L->set_coefficient("C_res_rho", C_res_rho);
  RES.L->set_coefficient("C_res_T", C_res_T);
  RES.L->set_coefficient("C_res_u", C_res_u);
}

void VD_NonlinearViscosity::assemble_system() {
  assemble(RES.A, *RES.a);
  coefficients_T_residual_.insert({"Norm_T_u_inf", Norm_T_u});
  coefficients_S_residual_.insert({"Norm_S_u_inf", Norm_S_u});
  RES.L->set_some_coefficients(coefficients_T_residual_);
}

void VD_NonlinearViscosity::compute_normalization() {

  int const global_jump = 0;

  *Fmom_loc->vector() = *Fvloc->vector();
  *Fmom_loc->vector() *= *Frho->vector();

  *FT_u_loc->vector() = *Fvloc->vector();
  *FT_u_loc->vector() *= *FT->vector();

  *FS_u_loc->vector() = *Fvloc->vector();
  *FS_u_loc->vector() *= *FS->vector();

  *Fkinetic_loc->vector() = *Fmom_loc->vector();
  *Fkinetic_loc->vector() *= *Fvloc->vector();

  mom_inf = std::max(Fmom_loc->vector()->max(), mom_inf);
  T_u_inf = std::max(FT_u_loc->vector()->max(), T_u_inf);
  S_u_inf = std::max(FS_u_loc->vector()->max(), S_u_inf);

  u_global_jump = std::max(
      fabs(Fvloc->vector()->max() - Fvloc->vector()->min()), DOLFIN_EPS_LARGE);
  *Norm_u = u_global_jump;


    mom_global_jump =
        std::max(fabs(Fmom_loc->vector()->max() - Fmom_loc->vector()->min()),
                 DOLFIN_EPS_LARGE);

    T_u_global_jump =
        std::max(fabs(FT_u_loc->vector()->max() - FT_u_loc->vector()->min()),
                 DOLFIN_EPS_LARGE);
    S_u_global_jump =
        std::max(fabs(FS_u_loc->vector()->max() - FS_u_loc->vector()->min()),
                 DOLFIN_EPS_LARGE);
  *Norm_mom = mom_global_jump;
  *Norm_T_u = T_u_global_jump;
  *Norm_S_u = S_u_global_jump;


    kin_global_jump = std::max(
        fabs(Fkinetic_loc->vector()->max() - Fkinetic_loc->vector()->min()),
        DOLFIN_EPS_LARGE);
  *Norm_kinetic = kin_global_jump;
}

void VD_NonlinearViscosity::set_to_zero() {
  nr_time_steps = 0;
  Fmu_->vector()->zero();
  Fmu_max_->vector()->zero();
  Fmu_low_->vector()->zero();
  Fmu_no_h_->vector()->zero();
  Fres_u_->vector()->zero();
  Fres_T_->vector()->zero();
  Fres_S_->vector()->zero();
  Fres_rho_->vector()->zero();
}

void VD_NonlinearViscosity::initialize_main_loop() { nr_time_steps = 0; }

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::compute() {
  set_log_active(false);
  tic();
  double const c_max = parameters["C_max"];
  int const constant_density = parameters["constant_density"];
  std::string const demo_type = parameters["demo_type"];

  if (c_max == 0.0) {
    return;
  }


  compute_normalization();

    *C_res_u = 1.0;
    *C_res_rho = 0.0;
    *C_res_T = 0.0;
    if (demo_type != "advection") {
    assemble(RES.b, *RES.L);
    ls_->solve(RES.A, *Fres_u_->vector(), RES.b);
    }


    if (!constant_density) {
      *C_res_rho = 1.0;
      *C_res_T = 0.0;
      *C_res_u = 0.0;
      assemble(RES.b, *RES.L);
      ls_->solve(RES.A, *Fres_rho_->vector(), RES.b);
    }

    if (!constantTemperature) {
      *C_res_rho = 0.0;
      *C_res_T = 1.0;
      *C_res_u = 0.0;
      RES.L->set_some_coefficients(coefficients_T_residual_);
      assemble(RES.b, *RES.L);
      ls_->solve(RES.A, *Fres_T_->vector(), RES.b);
      // change coefficients to S
      RES.L->set_some_coefficients(coefficients_S_residual_);
      assemble(RES.b, *RES.L);
      ls_->solve(RES.A, *Fres_S_->vector(), RES.b);
    }

  Fres_u_->vector()->abs();
  Fres_rho_->vector()->abs();
  Fres_T_->vector()->abs();
  Fres_S_->vector()->abs();

  compute_viscosity();

  set_log_active(true);
  info("RV computation: = %f sec", toc());
  set_log_active(false);
}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::compute_BDF1(Function const &S, Function const &S0,
                                         double const &kn, Function &dtS) {
  *dtS.vector() = *S.vector();
  *dtS.vector() -= *S0.vector();
  *dtS.vector() *= 1.0 / kn;
}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::compute_BDF2(Function const &S, Function const &S0,
                                         Function const &S00, double const &kn,
                                         double const &kn0, Function &dtS) {
  double dt1 = kn;
  double dt2 = kn + kn0;
  double C1 = (dt2 * dt2 - dt1 * dt1) / (dt1 * dt2);
  double C2 = -dt2 / dt1;
  double C3 = dt1 / dt2;

  *dtS.vector() = 0.;
  dtS.vector()->axpy(C1, *(S.vector()));
  dtS.vector()->axpy(C2, *(S0.vector()));
  dtS.vector()->axpy(C3, *(S00.vector()));
  *dtS.vector() *= 1.0 / (dt2 - dt1);
}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::compute_BDF3(Function const &S, Function const &S0,
                                         Function const &S00,
                                         Function const &S000, double const &k,
                                         double const &k0, double const &k00,
                                         Function &dtS) {

  // Rt = (V-18/11*V_prev+9/11*V_prev2-2/11*V_prev3)*11/(6*dt);

  double const w_np2 = k / k0;
  double const w_np1 = k0 / k00;

  double const alfa_0 = -pow(w_np1, 3) * pow(w_np2, 2) * (1 + w_np2) /
                        ((1 + w_np1) * (1 + w_np1 + w_np1 * w_np2));

  double const alfa_1 = pow(w_np2, 2) * (w_np1 + 1 / (1 + w_np2));
  double const alfa_2 = -1 - w_np2 - w_np1 * w_np2 * (1 + w_np2) / (1 + w_np1);

  double const alfa_3 =
      1 + w_np2 / (1 + w_np2) + w_np1 * w_np2 / (1 + w_np1 * (1 + w_np2));

  // (alfa_3*S + alfa_2*S0 + alfa_1*S00 + alfa_0*S000)/(k)

  *dtS.vector() = *(S.vector());
  *dtS.vector() *= alfa_3;
  dtS.vector()->axpy(alfa_2, *(S0.vector()));
  dtS.vector()->axpy(alfa_1, *(S00.vector()));
  dtS.vector()->axpy(alfa_0, *(S000.vector()));
  *dtS.vector() /= k;
}

void VD_NonlinearViscosity::compute_BDF4(Function const &S, Function const &S0,
                                         Function const &S00,
                                         Function const &S000,
                                         Function const &S0000, double const &k,
                                         double const &k0, double const &k00,
                                         double const &k000, Function &dtS) {

  double const w_np3 = k / k0;
  double const w_np2 = k0 / k00;
  double const w_np1 = k00 / k000;

  double const A1 = 1 + w_np1 * (1 + w_np2);
  double const A2 = 1 + w_np2 * (1 + w_np3);
  double const A3 = 1 + w_np1 * A2;

  double const alfa_0 = (1 + w_np3) / (1 + w_np1) * A2 / A1 *
                        (pow(w_np1, 4) * pow(w_np2, 3) * pow(w_np3, 2)) / A3;

  double const alfa_1 =
      -pow(w_np2, 3) * pow(w_np3, 2) * (1 + w_np3) / (1 + w_np2) * A3 / A2;

  double const alfa_2 = w_np3 * (w_np3 / (1 + w_np3) +
                                 w_np2 * w_np3 * (A3 + w_np1) / (1 + w_np1));

  double const alfa_3 = -1 - w_np3 * (1 + w_np2 * (1 + w_np3) / (1 + w_np2) *
                                              (1 + w_np1 * A2 / A1));

  double const alfa_4 =
      1 + w_np3 / (1 + w_np3) + w_np2 * w_np3 / A2 + w_np1 * w_np2 * w_np3 / A3;

  // alfa_4*u + alfa_3*u_n + alfa_2*u_nm1 + alfa_1*u_nm2 + alfa_0*u_nm3

  *dtS.vector() = *(S.vector());
  *dtS.vector() *= alfa_4;
  dtS.vector()->axpy(alfa_3, *(S0.vector()));
  dtS.vector()->axpy(alfa_2, *(S00.vector()));
  dtS.vector()->axpy(alfa_1, *(S000.vector()));
  dtS.vector()->axpy(alfa_0, *(S0000.vector()));
  *dtS.vector() /= k;
}

void VD_NonlinearViscosity::extrapolation2(Function const &S,
                                           Function const &S0, double const &k,
                                           double const &k0, Function &S_ext) {

  double const w_np1 = k / k0;

  double const beta_0 = -w_np1;

  double const beta_1 = (1 + w_np1);

  // (1 + ω n+1 )f (U n+1 ) − ω n+1 f (U n )

  *S_ext.vector() = *(S.vector());
  *S_ext.vector() *= beta_1;
  S_ext.vector()->axpy(beta_0, *(S0.vector()));
}

void VD_NonlinearViscosity::extrapolation3(Function const &S,
                                           Function const &S0,
                                           Function const &S00, double const &k,
                                           double const &k0, double const &k00,
                                           Function &S_ext) {

  double const w_np2 = k / k0;
  double const w_np1 = k0 / k00;

  double const beta_0 = pow(w_np1, 2) * w_np2 * (1 + w_np2) / (1 + w_np1);

  double const beta_1 = -w_np2 * (1 + w_np1 * (1 + w_np2));

  double const beta_2 = (1 + w_np2) * (1 + w_np1 * (1 + w_np2)) / (1 + w_np1);

  // alfa_4*u + alfa_3*u_n + alfa_2*u_nm1 + alfa_1*u_nm2 + alfa_0*u_nm3

  *S_ext.vector() = *(S.vector());
  *S_ext.vector() *= beta_2;
  S_ext.vector()->axpy(beta_1, *(S0.vector()));
  S_ext.vector()->axpy(beta_0, *(S00.vector()));
}

void VD_NonlinearViscosity::extrapolation4(
    Function const &S, Function const &S0, Function const &S00,
    Function const &S000, double const &k, double const &k0, double const &k00,
    double const &k000, Function &S_ext) {

  double const w_np3 = k / k0;
  double const w_np2 = k0 / k00;
  double const w_np1 = k00 / k000;

  double const A1 = 1 + w_np1 * (1 + w_np2);
  double const A2 = 1 + w_np2 * (1 + w_np3);
  double const A3 = 1 + w_np1 * A2;

  double const beta_0 = -pow(w_np1, 3) * pow(w_np2, 2) * w_np3 * (1 + w_np3) /
                        (1 + w_np1) * A2 / A1;

  double const beta_1 = pow(w_np2, 2) * w_np3 * (1 + w_np3) / (1 + w_np2) * A3;

  double const beta_2 = -A2 * A3 * w_np3 / (1 + w_np1);

  double const beta_3 = w_np2 * (1 + w_np3) / (1 + w_np2) *
                        ((1 + w_np3) * (A3 + w_np1) + (1 + w_np1) / w_np2) / A1;

  // alfa_4*u + alfa_3*u_n + alfa_2*u_nm1 + alfa_1*u_nm2 + alfa_0*u_nm3

  *S_ext.vector() = *(S.vector());
  *S_ext.vector() *= beta_3;
  S_ext.vector()->axpy(beta_2, *(S0.vector()));
  S_ext.vector()->axpy(beta_1, *(S00.vector()));
  S_ext.vector()->axpy(beta_0, *(S000.vector()));
}

//-----------------------------------------------------------------------------
void VD_NonlinearViscosity::Linf_u(Function const &FS, double &S_inf) {
  double S_bar = 0.0;
  std::size_t const S_array_size = FS.vector()->local_size();
  std::vector<double> S_array(S_array_size);
  FS.vector()->get_local(S_array);
  std::size_t const S_global_size = FS.vector()->size();

  for (std::size_t i = 0; i < S_array_size; ++i) {
    S_bar += S_array[i];
  }
  double S_bar_global = 0.0;
  S_bar_global = MPI::all_reduce(mesh_->mpi_comm(), S_bar, MPI_SUM);
  S_bar = S_bar_global / static_cast<double>(S_global_size);

  // S_inf = FS.vector()->norm("linf") - S_bar;
  S_inf = 0.0;
  for (std::size_t i = 0; i < S_array_size; ++i) {
    S_inf = std::max(S_inf, std::fabs(S_array[i] - S_bar));
  }
  double S_inf_global = 0.0;
  S_inf_global = MPI::max(mesh_->mpi_comm(), S_inf);
  // S_inf = std::max(S_inf_global, 1.0);
}

// -----------------------------------------------------------------------------
void VD_NonlinearViscosity::compute_viscosity() {
  int const D = mesh_->topology().dim();
  double const c_max = parameters["C_max"];
  double const c_e = parameters["C_R"];

  int const pol_degree_V = parameters["PolDegree_V_M"];
  int const pol_degree_Q = parameters["PolDegree_Q"];
  // h needs to be scaled properly depending if it lives in V or Q
  double const h_factor_Q = ((double)pol_degree_V) / ((double)pol_degree_Q);

  double const gamma = parameters["rv_gamma"];

  std::vector<double> Fu_arr(dof_V_.size());
  std::vector<double> Fvloc_arr(dof_M_.size());
  std::vector<double> Fres_u_arr(dof_M_.size());
  std::vector<double> Fres_T_arr(dof_M_.size());
  std::vector<double> Fres_S_arr(dof_M_.size());
  std::vector<double> Fres_rho_arr(dof_M_.size());
  std::vector<double> Fmu_arr(dof_M_.size());
  std::vector<double> Fkappa_arr(dof_M_.size());
  std::vector<double> Fkappa_T_arr(dof_M_.size());
  std::vector<double> Fkappa_S_arr(dof_M_.size());
  std::vector<double> Fmu_p_arr(dof_M_.size());
  std::vector<double> Fmu_max_arr(dof_M_.size());
  std::vector<double> Fh_arr(dof_M_.size());

  Fvloc->vector()->get_local(&Fvloc_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_u_->vector()->get_local(&Fres_u_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_rho_->vector()->get_local(&Fres_rho_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_T_->vector()->get_local(&Fres_T_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_S_->vector()->get_local(&Fres_S_arr[0], dof_M_.size(), &dof_M_[0]);
  Fh_PX->vector()->get_local(&Fh_arr[0], dof_M_.size(), &dof_M_[0]);

  std::size_t num_cell_nodes =
      Fvloc->function_space()->dofmap()->max_element_dofs();

  std::size_t offset_M = 0;
  std::size_t offset_u = 0;
  std::size_t ii = 0;
  std::size_t ii_v = 0;
  for (CellIterator cell(*mesh_); !cell.end(); ++cell) {

    // h = sqrt((*cell).volume());

    for (std::size_t node = 0; node < num_cell_nodes; ++node, ++ii) {
      double h = Fh_arr[ii];
      /*      double unorm = 0.0;
            for (int d = 0; d < value_dim_u; ++d) {
              std::size_t uii = offset_u + num_cell_nodes * d + node;
              unorm += Fu_arr[uii] * Fu_arr[uii];
            }*/

      /// compute entropy and vloc
      // FS_arr[ii] = 0.5 * unorm;

      double vloc = Fvloc_arr[ii]; // sqrt(unorm);

      double res_u = Fres_u_arr[ii];
      double res_rho = Fres_rho_arr[ii];
      double res_T = Fres_T_arr[ii];
      double res_S = Fres_S_arr[ii];

      double const mu_E_no_h_K = c_e * fabs(res_u);
      double const mu_max_K = c_max * h * (fabs(vloc) + DOLFIN_EPS_LARGE);
      double const mu_gd_K = h * (fabs(vloc) + DOLFIN_EPS_LARGE);

      res_u = std::min(fabs(res_u), 1.0);
      res_T = std::min(fabs(res_T), 1.0);
      res_S = std::min(fabs(res_S), 1.0);
      res_rho = std::min(fabs(res_rho), 1.0);

      Fmu_arr[ii] = mu_max_K * res_u;
      Fkappa_arr[ii] = mu_max_K * res_rho * h_factor_Q;
      Fkappa_T_arr[ii] = mu_max_K * res_T * h_factor_Q;
      Fkappa_S_arr[ii] = mu_max_K * res_S * h_factor_Q;

      Fmu_p_arr[ii] = h * res_u;

      Fmu_max_arr[ii] = mu_gd_K;

      Fres_u_arr[ii] = res_u;
      Fres_rho_arr[ii] = res_rho;
      Fres_T_arr[ii] = res_T;
      Fres_S_arr[ii] = res_S;
    }
    offset_u += Fu->function_space()->dofmap()->max_element_dofs();
  }

  Fres_u_->vector()->set_local(&Fres_u_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_u_->vector()->apply("insert");
  Fres_rho_->vector()->set_local(&Fres_rho_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_rho_->vector()->apply("insert");
  Fres_T_->vector()->set_local(&Fres_T_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_T_->vector()->apply("insert");
  Fres_S_->vector()->set_local(&Fres_S_arr[0], dof_M_.size(), &dof_M_[0]);
  Fres_S_->vector()->apply("insert");
}


//------------------------------------------------------------------------------
void VD_NonlinearViscosity::compute_grad_phi_CG1(
    std::shared_ptr<const FunctionSpace> Q_CG1, Function &Fgrad) {

  // get process rank:
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::size_t D = Q_CG1->mesh()->topology().dim();
  // // create a temporary function
  Function Fv(Q_CG1);
  std::vector<dolfin::la_index> dof_Q_CG1_;
  DofMappings::get_all_dofs(Fv, dof_Q_CG1_);
  std::vector<dolfin::la_index> dof_DG0_;
  DofMappings::get_all_dofs(Fgrad, dof_DG0_);
  std::vector<double> Fv_arr(dof_Q_CG1_.size());
  std::vector<double> FDG0_arr(dof_DG0_.size());

  Fv.vector()->get_local(&Fv_arr[0], dof_Q_CG1_.size(), &dof_Q_CG1_[0]);
  Fgrad.vector()->get_local(&FDG0_arr[0], dof_DG0_.size(), &dof_DG0_[0]);

  // Vector to hold dof map for a cell
  int form_rank = 1; // 0 equal normal?
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);
  auto element = Fv.function_space()->element();
  std::vector<double> coordinate_dofs;
  boost::multi_array<double, 2> coordinates;
  std::size_t num_cell_nodes =
      Fv.function_space()->dofmap()->max_element_dofs() / Fv.value_size();

  std::vector<dolfin::la_index> cell_dofs(num_cell_nodes);

  std::size_t ii = 0;
  std::size_t i_cell = 0;
  std::size_t offset_DG0 = 0.0;

  for (CellIterator cell(*Fv.function_space()->mesh()); !cell.end();
       ++cell, ++i_cell) {
    cell->get_vertex_coordinates(coordinate_dofs);
    element->tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);
    auto cell_dofs = Fv.function_space()->dofmap()->cell_dofs(cell->index());

    auto dmap = Fv.function_space()->dofmap()->cell_dofs(cell->index());
    auto ownership_range = Fv.function_space()->dofmap()->ownership_range();
    dofs[0] = ArrayView<const dolfin::la_index>(dmap.size(), dmap.data());

    // num_cell_nodes should be the same as element->space_dimension()

    std::vector<double> vl2(num_cell_nodes, 0);
    std::vector<double> grad_K_max(D, 0);
    std::vector<double> grad_K_average(D, 0);

    // compute the basis derivative (for P1)
    for (std::size_t i = 0; i < num_cell_nodes; ++i, ++ii) {
      std::vector<double> dval(D);
      std::vector<double> x(D);
      for (int k = 0; k < D; ++k) {
        x[k] = coordinates[i][k];
      }

      element->evaluate_basis_derivatives(i, 1, dval.data(), x.data(),
                                          coordinate_dofs.data(), 0);

      double l2 = 0.0;
      for (int j = 0; j < D; ++j) {
        grad_K_max[j] = std::max(grad_K_max[j], fabs(dval[j]));

        grad_K_average[j] += fabs(dval[j]);
      }

      const auto gid = cell_dofs[i];
      auto var = Fv.function_space()->dofmap()->local_to_global_index(gid);
      if (Fv.vector()->owns_index(var)) {
        Fv_arr[ii] = vl2[i];
        Fv.vector()->set_local(&Fv_arr[ii], 1, &dof_Q_CG1_[ii]);
      }
    }

    // compute the average of the gradient
    for (int j = 0; j < D; ++j) {
      grad_K_average[j] /= num_cell_nodes;
    }

    for (int j = 0; j < D; ++j) {
      std::size_t grad_K_pos = offset_DG0 + 1 * j;
      // FDG0_arr[grad_K_pos] = grad_K_max[j];
      FDG0_arr[grad_K_pos] = grad_K_average[j];
    }

    offset_DG0 += Fgrad.function_space()->dofmap()->max_element_dofs();

  }

  Fgrad.vector()->set_local(&FDG0_arr[0], dof_DG0_.size(), &dof_DG0_[0]);
  Fgrad.vector()->apply(
      "insert");

  Fv.vector()->apply("insert");
  Fv.vector()->apply("add");
}
