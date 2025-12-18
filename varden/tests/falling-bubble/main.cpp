// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2021-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
#include <ctime>
#include <fstream>

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/la/solve.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>

#include "varden/solver/ufl/VD_Mixed_2D_P1P1.h"
#include "varden/solver/ufl/VD_Mixed_3D_P1P1.h"
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSETracerEquation.h>
//#include <varden/pcs/VD_NSESolver_PCS.h>
#include <varden/solver/VD_NSESolver.h>
//#include "Norms.h"

/// core
#include <ConfigurableParameters.h>
#include <SlipBC.h>

#include "Bubble.h"
#include <mpi.h>

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  fem_varden_flow::ConfigurableParameters config_params("parameters.json");
  Parameters solver_parameters =
      config_params.to_dolfin_parameters("solver_parameters");

  // Print log messages only from the root process in parallel
  // the followings are global Dolfin parameters
  parameters["std_out_all_processes"] = false;
  parameters["allow_extrapolation"] = true;
  parameters["refinement_algorithm"] = "plaza";
  // parameters["refinement_algorithm"] = "plaza_with_parent_facets";

  parameters["reorder_dofs_serial"] = false;

  // MESHING **************************

  // Load mesh from file
  // auto mesh = std::make_shared<Mesh>(argv[5]);
  // auto mesh = std::make_shared<Mesh>("mesh.xml");
  // *mesh = refine(*mesh);
  time_t tstart, tend;
  tstart = time(0);

  int const nPoints = solver_parameters["n_points"];
  int const Nx = nPoints; // 40;//20;//80;//80;
  int const Nz = nPoints; // 40;//20;//80;//80;
  int const Ny = 2 * Nx;  // 80;//320; // should be 8* (I think. Before it was
                          // 4*)
  double t = 0.0;

  // Point p0(xmin, ymin, zmin);
  Point p0(xmin, ymin);
  // Point p1(xmax, ymax, zmax);
  Point p1(xmax, ymax);
  // RectangleMesh = rec(p0, p1, Nx, Ny,"right/left");
  auto mesh = std::make_shared<RectangleMesh>(p0, p1, Nx, Ny, "right/left");
  // auto mesh = std::make_shared<BoxMesh>(p0, p1, Nx, Ny, Nz);
  // ********************************************************

  // File fmesh("mesh.pvd");
  // fmesh << *mesh;

  // Set parameter values
  auto rho_in = std::make_shared<IndataDensity>(t);
  // auto rho_in = std::make_shared<Constant>(1.0);

  // Create function spaces
  auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh,
                                                             solver_parameters);
  auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
      mesh, solver_parameters);
  auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
      mesh, solver_parameters);
  std::shared_ptr<FunctionSpace> V1;

  int const D = mesh->topology().dim();

  std::shared_ptr<Constant> u_in, f, f_man;
  std::shared_ptr<Constant> zeros;

  if (D == 2) {
    V1 = std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh);

    zeros = std::make_shared<Constant>(0, 0);

    f_man = std::make_shared<Constant>(0.0, 0.0);
    u_in = std::make_shared<Constant>(0.0, 0.0);
    f = std::make_shared<Constant>(0.0, -sqrt(g));

  } else if (D == 3) {
    V1 = std::make_shared<VD_Mixed_3D_P1P1::CoefficientSpace_u_n>(mesh);

    zeros = std::make_shared<Constant>(0, 0, 0);

    f_man = std::make_shared<Constant>(0.0, 0.0, 0.);
    u_in = std::make_shared<Constant>(0.0, 0.0, 0.);
    f = std::make_shared<Constant>(0.0, -sqrt(g), 0.);
  }

  // Define values for boundary conditions
  auto zero = std::make_shared<Constant>(0);
  auto rho_high = std::make_shared<Constant>(rho1);
  auto rho_low = std::make_shared<Constant>(rho2);
  // change to -g later-9.81

  // Define subdomains for boundary conditions
  // auto slip_domain = std::make_shared<SlipDomain>();
  auto complete_boundary = std::make_shared<CompleteBoundary>();

  auto top_wall = std::make_shared<TopWall>();
  auto bottom_wall = std::make_shared<BottomWall>();
  auto top_bottom = std::make_shared<TopBottom>();
  auto left_right = std::make_shared<LeftRight>();

  // Define boundary conditions
  DirichletBC noslip(V, u_in, left_right);
  DirichletBC noslip_instead_of_slip(V, u_in, complete_boundary);

  DirichletBC top_density(M, rho_high, top_wall);
  DirichletBC low_density(M, rho_low, bottom_wall);
  DirichletBC slip_manual_lr(V->sub(0), zero, left_right);
  DirichletBC slip_manual_tb(V->sub(1), zero, top_bottom);
  // DirichletBC slip_manual_lr_z(V->sub(2), zero, left_right_Z);

  auto bcu = std::make_shared<std::vector<DirichletBC *>>();
  bcu->push_back(&slip_manual_lr);
  bcu->push_back(&slip_manual_tb);
  auto bcp = std::make_shared<std::vector<DirichletBC *>>();
  auto bcrho = std::make_shared<std::vector<DirichletBC *>>();

  fem_varden_flow::SlipBC slip(V, V1, complete_boundary);

  auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
  // slip_bcs->push_back(&slip);

  auto slip_bc_marker = std::make_shared<SlipBC_marker>();
  auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

  // dolfin::list_linear_solver_methods();
  // dolfin::list_krylov_solver_preconditioners();
  // Easier to choose Re in solver parameters and then convert to nu.
  double Re = solver_parameters["nu"];
  // double nu = rho2*pow(d,3.0/2.0)*sqrt(g)/Re;
  double nu = 1.0 / Re;
  solver_parameters["nu"] = nu;

  fem_varden_flow::VD_NSESolver ns(mesh, *u_in, rho_in, zero, f, f_man, bcu, bcp, bcrho,
                           slip_bcs, slip_bc_marker, slip_bc_marker_fn, V, Q, M,
                           solver_parameters, t, 0);

  ns.adaptive_solve();

  MPI_Finalize();
  return 0;
}
