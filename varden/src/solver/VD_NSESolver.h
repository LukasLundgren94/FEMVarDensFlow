// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Copyright (c) 2025 Jonathan Wiskandt
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __VD_NSE_SOLVER_H
#define __VD_NSE_SOLVER_H

#include <string>

#include "PDESmooth.h"

#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/mesh/Cell.h>

///
namespace dolfin {
class Mesh;
class Function;
class Cell;
// class CellIterator;
class Expression;
class FunctionSpace;
class DofMap;
class DirichletBC;
class LinearSolver;
} // namespace dolfin

///
using dolfin::Cell;
using dolfin::CellIterator;
// using dolfin::Constant;
using dolfin::DirichletBC;
using dolfin::DofMap;
using dolfin::Expression;
using dolfin::Form;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::LinearSolver;
using dolfin::Mesh;
using dolfin::Parameters;
using dolfin::Variable;

///
#include "../common/VD_Divergence.h"
#include "../common/VD_Functionals.h"
#include "../common/VD_NonlinearViscosity.h"
#include "FileSaver.h"
#include "System.h"
#include "NodeNormal.h"
#include "SlipBC.h"
#include "FarFieldTracer.h"
#include "FrictionVelocity.h"
#include "ThreeEquations.h"

namespace fem_varden_flow {
// class System;
class SlipBC;
class DoffMappings;
class VD_NonlinearViscosity;
class VD_Functionals;

class VD_NSESolver : public Variable {
public:
  ///
  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
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
               Parameters &p, double &time, double dt);

  ///
  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
               std::shared_ptr<const Expression> initial_rho,
               std::shared_ptr<const Expression> initial_P,
               std::shared_ptr<const Expression> f,
               std::shared_ptr<const Expression> f_man,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
               std::shared_ptr<const FunctionSpace> V,
               std::shared_ptr<const FunctionSpace> Q,
               std::shared_ptr<const FunctionSpace> M, Parameters &p,
               double &time, double dt);

  ///
  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
               std::shared_ptr<const Expression> initial_rho,
               std::shared_ptr<const Expression> initial_P,
               std::shared_ptr<const Expression> f,
               std::shared_ptr<const Expression> f_man,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_u,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_P,
               std::shared_ptr<std::vector<DirichletBC *>> bcs_rho,
               Parameters &p, double &time, double dt);

  ///
  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
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
               std::shared_ptr<const FunctionSpace> M, Parameters &p,
               double &time, double dt);

  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
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
               std::shared_ptr<const FunctionSpace> M, Parameters &p,
               double &time, double dt);

  VD_NSESolver(std::shared_ptr<const Mesh> mesh, Expression const &initial_u,
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
               std::shared_ptr<MeshFunction<std::size_t>> sub_domains,
               double &time, double dt);

  // names are difficult:

  // f_S, f_T, f_r, and so on. what should I call all of the source functions?
  // f_source_T, f_source_S, f_reaction_S f_restore_T, f_source_T, f_melt,
  // f_restore, f_1_T, f_2_S.... so many options

  ///
  VD_NSESolver();

  ///
  ~VD_NSESolver();

  ///
  // void solve();

  ///
  void solve(std::size_t n_retreive = 0);

  void adaptive_solve(std::size_t n_retreive = 0);

  ///
  std::shared_ptr<Function> &get_Fu();

  std::shared_ptr<Function> &get_Frho();
  std::shared_ptr<Function> &get_FT();
  std::shared_ptr<Function> &get_FT_F();
  std::shared_ptr<Function> &get_FS();

  std::shared_ptr<double> get_T_bar();
  std::shared_ptr<double> get_S_bar();
  std::shared_ptr<double> get_rho_bar();

  std::shared_ptr<Function> &get_FP();

  std::shared_ptr<Function> &get_Fh();

  double get_mesh_volume();

  double get_final_dt();

  ///
  void clear();

  ///
  std::shared_ptr<Form> Form_uavg;
  std::shared_ptr<Form> Form_tau_w_l;
  std::shared_ptr<Form> Form_tau_w_u;

  std::shared_ptr<MeshFunction<std::size_t>> sub_domains_;
  std::shared_ptr<MeshFunction<std::size_t>> marker_bad_cells_;


  ///

protected:
  Parameters average_parameters;
  // Parameters parameters;
  int schur_solver;
  int grad_div_splitting;
  int saddle_point;
  int constant_density;
  int remove_pressure_average;
  int remove_rho_average;
  int initial_guess_extrapolation;
  int bad_cells_boundary_condition;
  int melt_with_virtual_flux;
  std::size_t dim_;
  double c_brezzi;
  double flux_h_factor;

  /// current time and time-step
  double &time_;
  double dt_, dt0_, dt00_, dt000_;
  double Tend_;
  double t_avg_start_;
  double mesh_volume_;
  double h_max;
  double s_cfl_;
  double s_u_;
  double s_rho_;
  double functional_offset_;
  double boundary_surface_;
  double dt_pid_prev_error_;
  bool dt_pid_has_prev_error_;
  double mass = 1.0;
  double mass_T = 1.0;
  double mass_S = 1.0;
  std::shared_ptr<double> rho_bar_;
  std::shared_ptr<double> T_bar_;
  std::shared_ptr<double> S_bar_;
  double h_PX_min = 0.0;
  double momentum_x = 0.0;
  double momentum_y = 0.0;
  double momentum_z = 0.0;
  double angular_momentum_z = 0.0;
  double kinetic_energy = 0.0;
  double maxDiv = 0.0;

  bool constantTemperature = true;
  bool project_uu;
  int melting_with_rv; // 0 = regular 3-equation, 1 = with tensor RV, 2 = with
                       // scalar RV
  std::map<std::string, std::shared_ptr<const GenericFunction>>
      coefficients_rho_u;
  std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_T;
  std::map<std::string, std::shared_ptr<const GenericFunction>>
      coefficients_only_T;
  std::map<std::string, std::shared_ptr<const GenericFunction>>
      coefficients_only_P;

  std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_S;
  std::map<std::string, std::shared_ptr<const GenericFunction>>
      coefficients_S_residual;
  std::map<std::string, std::shared_ptr<const GenericFunction>>
      coefficients_only_S;

  std::shared_ptr<PDESmooth> PDESmooth_rho_;

  int num_iter_mom0, num_iter_rho, num_iter_T0;

  void get_dt_cfl(Function const &Fvloc,
                  std::vector<dolfin::la_index> const &dof_Q, double const &t,
                  double &k, Parameters const &parameters);

  void adapt_dt(Function const &Fvloc,
                std::vector<dolfin::la_index> const &dof_Q, double const &t,
                double &k, Parameters const &parameters, bool &redo_step);

  void get_rho_min(Function const &Frho,
                   std::vector<dolfin::la_index> const &dof_Q, double &rho_min);

  ///
  void compute_unknowns();

  void compute_initial(bool is_checkpoint);

  void compute_UU();

  void compute_LPS();

  void compute_outflow_pressure();

  void compute_solutions();

  void compute_time_derivatives();

  void compute_statistics();

  void redo_compute_solutions();

  void solve_systems();

  void compute_melting();

  void
  interpolate_from_manifold_to_domain(std::shared_ptr<Function> F_domain_,
                                      std::shared_ptr<Function> F_manifold_);

  void first_order_iteration();
  void second_order_iteration();
  void third_order_iteration();
  void fourth_order_iteration();
  void time_step_iteration(int time_order);

  void constant_dt_init();

  std::shared_ptr<PETScMatrix> createDiagonalMatrix(const GenericVector &diag,
                                                    const GenericMatrix &A,
                                                    const Mesh &mesh);

  double assemble_functional(const Form &a, Scalar &s);

  std::shared_ptr<Scalar> scalar;

  std::shared_ptr<Constant> Ck;
  std::shared_ptr<Constant> Ck0;
  std::shared_ptr<Constant> Ck00;
  std::shared_ptr<Constant> Ck000;
  std::shared_ptr<Constant> C_BDF1_;
  std::shared_ptr<Constant> C_BDF3_;
  std::shared_ptr<Constant> C_BDF4_;
  std::shared_ptr<Constant> C_crank_nicolson_;
  std::shared_ptr<Constant> e_x_;
  std::shared_ptr<Constant> e_y_;
  std::shared_ptr<Constant> e_z_;
  std::shared_ptr<Constant> Cnu_;
  std::shared_ptr<Constant> Cprandtl;
  std::shared_ptr<Constant> Clambda;
  std::shared_ptr<Constant> Clambda_grad_div;
  std::shared_ptr<Constant> Clambda_grad_div_u;
  std::shared_ptr<Constant> Cgamma_grad_div;
  std::shared_ptr<Constant> Crho_inf;
  std::shared_ptr<Constant> Cvloc_inf;
  std::shared_ptr<Constant> C_mass;
  std::shared_ptr<Constant> C_laplace;
  std::shared_ptr<Constant> C_brezzi_;
  std::shared_ptr<Constant> Cslip_penalty;
  std::shared_ptr<Constant> Cslip_penalty_fn;
  std::shared_ptr<Constant> Crv_smoothing_weight;
  std::shared_ptr<Constant> C_newton_;
  std::shared_ptr<Constant> C_skew_factor_;
  std::shared_ptr<Constant> C_skew_rho_factor_;
  std::shared_ptr<Constant> C_skew_T_factor_;
  std::shared_ptr<Constant> C_skew_S_factor_;
  std::shared_ptr<Constant> C_div_factor_;
  std::shared_ptr<Constant> C_mom_update_;
  std::shared_ptr<Constant> C_mom_avg_;
  std::shared_ptr<Constant> C_shift_factor_;
  std::shared_ptr<Constant> C_rho_mass_;
  std::shared_ptr<Constant> C_T_mass_;
  std::shared_ptr<Constant> C_S_mass_;
  std::shared_ptr<Constant> C_ddn1_;
  std::shared_ptr<Constant> C_ddn2_;
  std::shared_ptr<Constant> C_max_;
  std::shared_ptr<Constant> C_max_vms_;
  std::shared_ptr<Constant> C_directional_wave_speed_;
  std::shared_ptr<Constant> C_omega_;
  std::shared_ptr<Constant> C_one_;
  std::shared_ptr<Constant> C_zero_;
  std::shared_ptr<Constant> C_zeros_;
  std::shared_ptr<Constant> C_project_uu_;
  std::shared_ptr<Constant> C_nitsche_penalty_;

  std::shared_ptr<Constant> C_nu_sym_;
  std::shared_ptr<Constant> C_mu_rho_;
  std::shared_ptr<Constant> C_kappa_;
  std::shared_ptr<Constant> C_kappa_T_;
  std::shared_ptr<Constant> C_kappa_S_;
  std::shared_ptr<Constant> C_eps_norm_;

  std::shared_ptr<Function> Frho_inf_K_;
  std::shared_ptr<Function> Fmom_K_;
  std::shared_ptr<Function> Frho_inf_patch_;

  ///
  std::shared_ptr<Function> Fmu_fine_;
  std::shared_ptr<Function> Fres_u_fine_;
  std::shared_ptr<Function> Fres_T_fine_;
  std::shared_ptr<Function> Fres_S_fine_;
  std::shared_ptr<Function> Fres_rho_fine_;
  std::shared_ptr<Function> Fdtu_;
  std::shared_ptr<Function> Fgrad_S_;
  std::shared_ptr<Function> Fgrad_T_;
  std::shared_ptr<Function> Fgrad_rho_;
  std::shared_ptr<Function> Fgrad_u_;
  std::shared_ptr<Function> Fgrad_p_;
  std::shared_ptr<Function> Fgrad_phi_DG0_;
  std::shared_ptr<Function> F_p_div_u_;
  std::shared_ptr<Function> Fu_fine_;
  std::shared_ptr<Function> Frho_fine_;
  std::shared_ptr<Function> FT_fine_;
  std::shared_ptr<Function> FS_fine_;
  std::shared_ptr<Function> Frho_exact_;
  std::shared_ptr<Function> Frho_t_;
  std::shared_ptr<Function> FT_t_;
  std::shared_ptr<Function> FS_t_;
  std::shared_ptr<Function> Fu_t_;
  std::shared_ptr<Function> Fvloc_;
  std::shared_ptr<Function> Fvloc_x_;
  std::shared_ptr<Function> Fvloc_y_;
  std::shared_ptr<Function> Fvloc_z_;
  std::shared_ptr<Function> Fmom_;
  std::shared_ptr<Function> FP_outflow_;
  std::shared_ptr<Function> Fh_PX_;
  std::shared_ptr<Function> F_one_;
  std::shared_ptr<Function> F_uu_projected_;

  std::shared_ptr<Function> Fu_boundary_;

  std::shared_ptr<Function> Frho_np1_;
  std::shared_ptr<Function> Frho_n_;
  std::shared_ptr<Function> Frho_nm1_;
  std::shared_ptr<Function> Frho_nm2_;
  std::shared_ptr<Function> Frho_nm3_;

  std::shared_ptr<Function> FS_np1_;
  std::shared_ptr<Function> FS_n_;
  std::shared_ptr<Function> FS_nm1_;
  std::shared_ptr<Function> FS_nm2_;
  std::shared_ptr<Function> FS_nm3_;

  std::shared_ptr<Function> FT_np1_;
  std::shared_ptr<Function> FT_n_;
  std::shared_ptr<Function> FT_nm1_;
  std::shared_ptr<Function> FT_nm2_;
  std::shared_ptr<Function> FT_nm3_;

  std::shared_ptr<Function> F_flux_S_;
  std::shared_ptr<Function> F_flux_T_;

  std::shared_ptr<Function> F_UU_;

  std::shared_ptr<Function> Frho_bar_;
  std::shared_ptr<Function> FT_bar_;
  std::shared_ptr<Function> FS_bar_;

  std::shared_ptr<Function> Fu_np1_;
  std::shared_ptr<Function> Fu_n_;
  std::shared_ptr<Function> Fu_nm1_;
  std::shared_ptr<Function> Fu_nm2_;
  std::shared_ptr<Function> Fu_nm3_;

  std::shared_ptr<Function> FP_np1_;
  std::shared_ptr<Function> FP_n_;
  std::shared_ptr<Function> FP_nm1_;
  std::shared_ptr<Function> FP_nm2_;
  std::shared_ptr<Function> FP_nm3_;

  std::shared_ptr<Function> Fdiag_;
  std::shared_ptr<Function> Fdiag_I_;

  std::shared_ptr<Function> Fdiv_;
  std::shared_ptr<Function> Fdiv_fine_;

  ///
  std::shared_ptr<Function> Fux_;
  std::shared_ptr<Function> Fuy_;
  std::shared_ptr<Function> Fuz_;

  std::shared_ptr<Function> Fu_mean_;
  std::shared_ptr<Function> Fv_mean_;
  std::shared_ptr<Function> Fw_mean_;

  std::shared_ptr<Function> Fuu_rey_;
  std::shared_ptr<Function> Fuv_rey_;
  std::shared_ptr<Function> Fuw_rey_;

  std::shared_ptr<Function> Fvv_rey_;
  std::shared_ptr<Function> Fvw_rey_;
  std::shared_ptr<Function> Fww_rey_;

  std::shared_ptr<Function> Fu_avg_;
  std::shared_ptr<Function> FT_avg_;
  std::shared_ptr<Function> FS_avg_;
  std::shared_ptr<Function> Fmelt_avg_;
  std::shared_ptr<Function> Frho_b_avg_;

  std::shared_ptr<Function> Fslipbc_marker_;
  std::shared_ptr<Function> Fslipbc_marker_fn_;
  std::shared_ptr<Function> Fnormal_;
  std::shared_ptr<Function> Ftau1_;
  std::shared_ptr<Function> Ftau2_;

  ///
  System TRACER_EQ_;
  System PRESS_FIRST_EQ_;

  System MOM_EQ_;
  System PRESS_EQ_;
  System PRESS_INITIAL_EQ_;
  System PRESS_EQ2_;
  System PRESS_EQ3_;
  System PRESS_EQ4_;
  System DIV_FORM_;
  System OUTFLOW_EQ_;
  System H_EQ_;

  System VLOC_EQ_;
  System LPS_EQ_;
  System LPS_TENSOR_EQ_;

  // std::shared_ptr<Matrix> A_mom;
  std::shared_ptr<Matrix> A_con_;
  std::shared_ptr<Matrix> A_press_;

  std::shared_ptr<Vector> b_con_;
  std::shared_ptr<Vector> b_press_;

  std::shared_ptr<Matrix> M_P1_fine;

  ///
  std::shared_ptr<const Mesh> mesh_;
  std::shared_ptr<Mesh> mesh_fine_;

  Expression const &initial_u;
  std::shared_ptr<const Expression> initial_rho_;
  std::shared_ptr<const Expression> initial_T_;
  std::shared_ptr<const Expression> initial_S_;
  std::shared_ptr<const Expression> initial_P_;
  std::shared_ptr<const Expression> f_T_;
  std::shared_ptr<const Expression> f_S_;
  std::shared_ptr<const Expression> f_source_T_;
  std::shared_ptr<const Expression> f_reaction_T_;
  std::shared_ptr<const Expression> f_source_S_;
  std::shared_ptr<const Expression> f_reaction_S_;

  std::shared_ptr<const Expression> slipbc_marker_;
  std::shared_ptr<const Expression> slipbc_marker_fn_;

  std::shared_ptr<const Expression> f;
  std::shared_ptr<const Expression> f_man;

  std::shared_ptr<std::vector<DirichletBC *>> bcs_u_;
  std::shared_ptr<std::vector<DirichletBC *>> bcs_P_;
  std::shared_ptr<std::vector<DirichletBC *>> bcs_rho_;
  std::shared_ptr<std::vector<DirichletBC *>> bcs_T_;
  std::shared_ptr<std::vector<DirichletBC *>> bcs_S_;

  std::shared_ptr<const std::vector<SlipBC *>> slip_bcs_;
  std::shared_ptr<fem_varden_flow::NodeNormal> unit_node_normal_;

  ///
  std::shared_ptr<const FunctionSpace> V_;
  std::shared_ptr<const FunctionSpace> Q_;
  std::shared_ptr<const FunctionSpace> M_;
  std::shared_ptr<const FunctionSpace> DG0V_;
  std::shared_ptr<FunctionSpace> M_fine;
  std::shared_ptr<FunctionSpace> V_fine;
  std::shared_ptr<FunctionSpace> Q_CG1_;

  ///
  std::vector<dolfin::la_index> dof_V_;
  std::vector<dolfin::la_index> dof_Q_;
  std::vector<dolfin::la_index> dof_M_;
  std::vector<dolfin::la_index> dof_DG0_;
  std::vector<dolfin::la_index> dof_DG0_V_;

  ///
  std::shared_ptr<LinearSolver> lsm_;
  std::shared_ptr<LinearSolver> lsP_;
  std::shared_ptr<LinearSolver> lsrho_;
  std::shared_ptr<LinearSolver> ls_mass_;

  ///
  std::shared_ptr<VD_NonlinearViscosity> NV_;
  std::shared_ptr<VD_Functionals> FUNC_;

  std::shared_ptr<PETScKrylovSolver> schur_krylovsolver;
  std::shared_ptr<PETScMatrix> A_nest;
  std::shared_ptr<PETScVector> sol_vec;
  std::shared_ptr<PETScVector> sol_vec1;
  std::shared_ptr<PETScVector> sol_vec2;
  std::shared_ptr<PETScVector> sol_vec3;

  std::shared_ptr<PETScVector> rhs_vec;

  std::shared_ptr<Form> a00;
  std::shared_ptr<Form> a01;
  std::shared_ptr<Form> a10;
  std::shared_ptr<Form> a11;

  std::shared_ptr<Form> a10_mom0;
  std::shared_ptr<Form> a10_mom1;
  std::shared_ptr<Form> a10_mom2;

  std::shared_ptr<Matrix> A00;
  std::shared_ptr<Matrix> A01;
  std::shared_ptr<Matrix> A10;
  std::shared_ptr<Matrix> A11;
  std::shared_ptr<PETScMatrix> Diagonal_M_L;
  std::shared_ptr<PETScMatrix> Diagonal_M_L_I;

  std::shared_ptr<Matrix> M_L;
  std::shared_ptr<Matrix> ASchur;
  std::shared_ptr<Form> aSchur;

  std::shared_ptr<Form> a_flux;
  std::shared_ptr<Form> L_flux;
  std::shared_ptr<Matrix> A_flux;
  std::shared_ptr<Vector> b_flux;

  std::shared_ptr<Vector> b0;
  std::shared_ptr<Vector> b1;

  ///
  std::shared_ptr<FileSaver> file_saver_;
  std::shared_ptr<FileSaver> file_saver_fine_;
  std::shared_ptr<FileSaver> file_saver_fine_average_;

  // melting_info
  System MANIFOLD_EQ_;

  std::shared_ptr<Function> Frho_boussinesq_;
  double avg_melt_rate_;
  double tau_ds_;
  double tau_dx_;
  double boundary_layer_thickness;
  double rho_T_factor;
  double rho_S_factor;
  double rho_boussinesq_shift;
  std::shared_ptr<Constant> C_volume_;
  std::shared_ptr<Constant> C_slip_friction_;
  std::shared_ptr<Constant> Cmelt_smoothing_weight_;
  std::shared_ptr<Function> Fmelt_manifold_;
  std::shared_ptr<Function> F_node_normal_manifold_;
  std::shared_ptr<Function> Fu_manifold_;

  std::shared_ptr<fem_varden_flow::FarFieldTracer> ff_S;
  std::shared_ptr<fem_varden_flow::FarFieldTracer> ff_T;
  std::shared_ptr<fem_varden_flow::FrictionVelocity> friction_vel;
  std::shared_ptr<fem_varden_flow::ThreeEquations> three_eq;
  std::shared_ptr<fem_varden_flow::ExchangeVelocity> exchangevelocity;
  std::shared_ptr<dolfin::DirichletBC> melt_vector_bc;
  std::shared_ptr<dolfin::DirichletBC> melt_vector_bc0;
  std::shared_ptr<dolfin::DirichletBC> melt_vector_bc1;
  std::shared_ptr<dolfin::DirichletBC> melt_vector_bc2;
  std::shared_ptr<dolfin::DirichletBC::Map> boundary_vals_v;
  std::shared_ptr<dolfin::DirichletBC::Map> boundary_vals_m;
  int size_m;
  int size_v;
  std::shared_ptr<std::vector<dolfin::la_index>> boundary_dofs_v;
  std::shared_ptr<std::vector<dolfin::la_index>> boundary_dofs_m;
  std::shared_ptr<std::vector<double>> M_dof_coors;
  std::shared_ptr<Function> F_melting_flux_S_;
  std::shared_ptr<Function> F_melting_flux_T_;
  std::shared_ptr<Function> F_melt_;
  std::shared_ptr<Function> F_melt_V_;
  std::shared_ptr<Function> F_melt_VV_;
  std::shared_ptr<Function> FT_b_;
  std::shared_ptr<Function> FS_b_;
  std::shared_ptr<FunctionSpace> V_P1_;
  std::shared_ptr<DirichletBC> melt_DBC_T_;
  std::shared_ptr<DirichletBC> melt_DBC_S_;
  std::shared_ptr<SlipBC> inflow_bc_melt_;

  double mean_T, mean_S, mean_mb = 1.0e7;
  std::size_t number_of_averages_ = 0;
  // new members for GammaT
  std::shared_ptr<dolfin::Function> F_gammaT_;
  std::shared_ptr<dolfin::Function> F_gammaS_;

  void init();

  void init_temperature();

  void init_ocean();

  void store_previous_solutions();

  void relaxation_conservation_fix();

  void add_noslip_on_bad_cells();

  /// compute volume of the mesh
  inline double volume(Mesh const &mesh) {
    double volume_l = 0.0;
    for (CellIterator cell(mesh); !cell.end(); ++cell) {
      volume_l += cell->volume();
    }
    double const volume_g = dolfin::MPI::sum(mesh.mpi_comm(), volume_l);
    return volume_g;
  };
};
} // namespace fem_varden_flow

// namespace fem_varden_flow

#endif
