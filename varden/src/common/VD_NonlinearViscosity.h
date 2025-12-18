// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __N_VD_VISCOSITY_H
#define __N_VD_VISCOSITY_H

#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>

#include "System.h"

///
namespace dolfin {
 class DofMap; 
 class Function; 
 class FunctionSpace; 
 class Mesh; 
 class SubDomain; 
 class Form; 
 class GenericMatrix; 
 class GenericVector; 
class DirichletBC;
 class Matrix; 
 class Vector; 
 class BoundaryNormal; 
 class BoundaryMesh; 
 template<typename T> class MeshFunction; 
} // namespace dolfin

///
 using dolfin::DofMap; 
 using dolfin::Function; 
 using dolfin::FunctionSpace; 
 using dolfin::Mesh; 
 using dolfin::BoundaryMesh; 
 using dolfin::SubDomain; 
 using dolfin::Form; 
 using dolfin::GenericMatrix; 
 using dolfin::GenericVector; 
using dolfin::CellIterator; 
 using dolfin::VertexIterator; 
 using dolfin::MeshFunction; 
 using dolfin::Hierarchical; 
 using dolfin::Variable; 
 using dolfin::Parameters; 
 using dolfin::Matrix; 
 using dolfin::Vector; 
 using dolfin::DirichletBC; 
 using dolfin::File; 
 using dolfin::Array; 
using dolfin::Constant;
 using dolfin::Cell; 
 using dolfin::SubDomain; 

///
namespace fem_varden_flow {
class VD_NonlinearViscosity // : public Variable
{
public:
  ///
  VD_NonlinearViscosity(std::shared_ptr<const Mesh> mesh,
                        std::vector<dolfin::la_index> &dof_M,
                        std::vector<dolfin::la_index> &dof_V,
                        Parameters &parameters);

  VD_NonlinearViscosity(std::shared_ptr<const Mesh> mesh,
                        std::shared_ptr<const FunctionSpace> M,
                        std::vector<dolfin::la_index> &dof_M,
                        std::vector<dolfin::la_index> &dof_V,
                        Parameters &parameters);

  ///
  ~VD_NonlinearViscosity() = default;

  void assemble_system();

  ///
  void compute();



  ///
  void compute_projected_dt();

  ///
  void compute_BDF2(Function const &S, Function const &S0, Function const &S00,
                    double const &kn, double const &kn0, Function &dtu);

  void compute_BDF3(Function const &S, Function const &S0, Function const &S00,
                    Function const &S000, double const &k, double const &k0,
                    double const &k00, Function &dtS);

  void compute_BDF4(Function const &S, Function const &S0, Function const &S00,
                    Function const &S000, Function const &S0000,
                    double const &k, double const &k0, double const &k00,
                    double const &k000, Function &dtS);
  ///

  void extrapolation4(Function const &S, Function const &S0,
                      Function const &S00, Function const &S000,
                      double const &k, double const &k0, double const &k00,
                      double const &k000, Function &S_ext);

  void extrapolation3(Function const &S, Function const &S0,
                      Function const &S00, double const &k, double const &k0,
                      double const &k00, Function &S_ext);

  void extrapolation2(Function const &S, Function const &S0, double const &k,
                      double const &k0, Function &S_ext);

  ///
  void compute_BDF1(Function const &S, Function const &S0, double const &kn,
                    Function &dtu);

  ///
  void compute_mesh_function(std::shared_ptr<const Mesh> mesh, Function &Fh);

  void set_to_zero();

  void initialize_main_loop();

  void compute_grad_phi(Function& Fgrad, Function& Flump);

  void compute_grad_phi_CG1(
					      std::shared_ptr<const FunctionSpace> Q_CG1,
					      Function& Fgrad);

  ///
  std::shared_ptr<Function> &get_res_u() { return Fres_u_; }
  std::shared_ptr<Function> &get_res_T() { return Fres_T_; }
  std::shared_ptr<Function> &get_res_S() { return Fres_S_; }
  std::shared_ptr<Function> &get_res_rho() { return Fres_rho_; }

  ///
  System RES, RES1;

  ///
  std::shared_ptr<Function> Fu;
  // std::shared_ptr<Function> Fmom;
  std::shared_ptr<Function> FS;
  std::shared_ptr<Function> FP;
  std::shared_ptr<Function> FT;
  std::shared_ptr<Function> Frho;
  std::shared_ptr<Function> Fvloc;
  std::shared_ptr<Function> Fmom_loc;
  std::shared_ptr<Function> Fkinetic_loc;
  std::shared_ptr<Function> FT_u_loc;
  std::shared_ptr<Function> FS_u_loc;
  std::shared_ptr<Function> Fh_PX;

  std::vector<std::vector<const DirichletBC *>> bcs_u_;

  std::vector<std::vector<const DirichletBC *>> bcs_res_;

  std::shared_ptr<Function> Fvloc_tilde;

  std::shared_ptr<Constant> Norm_mom;
  std::shared_ptr<Constant> Norm_T_u;
  std::shared_ptr<Constant> Norm_S_u;
  std::shared_ptr<Constant> Norm_u;
  std::shared_ptr<Constant> Norm_rho;
  std::shared_ptr<Constant> Norm_kinetic;

  std::shared_ptr<Constant> Cthreshold;
  std::shared_ptr<Constant> Cgamma;
  std::shared_ptr<Constant> C_res_u;
  std::shared_ptr<Constant> C_res_rho;
  std::shared_ptr<Constant> C_res_T;
  std::shared_ptr<Constant> C_res_S;
  std::shared_ptr<Constant> C_R;

  double norm_rho = 1.0;

  double T_u_inf = 1.0;
  double S_u_inf = 1.0;
  double mom_inf = 1.0;

  double u_global_jump= 0.0;
  double mom_global_jump = 0.0;
  double T_u_global_jump = 0.0;
  double S_u_global_jump = 0.0;
  double kin_global_jump = 0.0;

  bool constantTemperature = false;



  std::shared_ptr<Function> Fres_u_fine_;
  std::shared_ptr<Function> Fres_rho_fine_;
  std::shared_ptr<Function> Fres_T_fine_;
  std::shared_ptr<Function> Fres_S_fine_;

  std::map<std::string, std::shared_ptr<const dolfin::GenericFunction>> coefficients_S_residual_;
  std::map<std::string, std::shared_ptr<const dolfin::GenericFunction>> coefficients_T_residual_;
  ///
  Matrix M;

private:

  ///
  void init();
  void configure_residual(const std::shared_ptr<const FunctionSpace>& M);

  ///
  void Linf_u(Function const &u_, double &u_inf_);

  ///
  void compute_viscosity();

  void compute_normalization();


  void maximum_on_patch(Function const &F, Function &Foutput,
                        Matrix const Mass_matrix);

  std::vector<double> grad_phi_m(boost::multi_array<double, 2> const & coordinates,
				 int const & v_i,
				 int const & v_j,
				 int const & v_k);



  class CompleteBoundary : public SubDomain {
    bool inside(const Array<double> &x, bool on_boundary) const {
      return on_boundary;
    }
  };

  ///
  std::vector<dolfin::la_index> &dof_V_;
  std::vector<dolfin::la_index> &dof_M_;

  std::vector<dolfin::la_index> dof_DG0_;
  std::vector<dolfin::la_index> dof_V_DG0_;

  ///
  std::shared_ptr<const Mesh> mesh_;
  Parameters &parameters;

  ///
  std::shared_ptr<Function> Fres_u_;
  std::shared_ptr<Function> Fres_T_;
  std::shared_ptr<Function> Fres_S_;
  std::shared_ptr<Function> Fres_rho_;

  std::shared_ptr<Function> Fmu_;

  std::shared_ptr<Function> Fmu_max_;
  std::shared_ptr<Function> Fmu_low_;
  std::shared_ptr<Function> Fmu_no_h_;

  std::shared_ptr<Function> Fmu1_;
  std::shared_ptr<Function> Fmu2_;
  std::shared_ptr<Function> Fmu3_;

  std::shared_ptr<dolfin::LinearSolver> ls_;
  std::shared_ptr<Function> FdtS_P_;

  std::shared_ptr<DirichletBC> bc;

  int nr_time_steps = 0;
};

} // namespace fem_varden_flow
#endif
