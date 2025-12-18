// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/la/solve.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>

/// varden
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSESolver.h>
#include <varden/solver/VD_NSETracerEquation.h>

/// core
#include <SlipBC.h>

#include "Buoyancy.h"

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  fem_varden_flow::ConfigurableParameters config_params("parameters.json");
  Parameters solver_parameters =
      config_params.to_dolfin_parameters("solver_parameters");

  parameters["std_out_all_processes"] = false;
  parameters["allow_extrapolation"] = true;
  parameters["refinement_algorithm"] = "plaza";
  parameters["reorder_dofs_serial"] = false;

  std::cout << "Start:" << std::endl;

  const int pol_degree_V_M = solver_parameters["PolDegree_V_M"];
  const int pol_degree_Q = solver_parameters["PolDegree_Q"];
  const int n_points = solver_parameters["n_points"];

  const int Nx = n_points;
  const int Ny = Nx;

  Point p0(xmin, ymin);
  Point p1(xmax, ymax);

  tic();
  auto mesh = std::make_shared<Mesh>(RectangleMesh::create(
      {{p0, p1}},
      {{static_cast<std::size_t>(Nx), static_cast<std::size_t>(Ny)}},
      CellType::Type::triangle, "right/left"));
  info("Mesh generation took %lf seconds", toc());

  double t = 0.0;
  double dt = 0.0;

  auto rho_in = std::make_shared<IndataDensity>(t);
  auto T_in = std::make_shared<IndataTemperature>(t);

  auto u_in = std::make_shared<Constant>(0.0, 0.0);
  auto zero = std::make_shared<Constant>(0.0);
  auto f = std::make_shared<Constant>(0.0, 0.0);
  auto f_man = std::make_shared<Constant>(0.0, 0.0);

  const double Ra = solver_parameters["rayleigh_number"];
  auto f_T = std::make_shared<Constant>(0.0, Ra);

  auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh,
                                                             solver_parameters);
  auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
      mesh, solver_parameters);
  auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
      mesh, solver_parameters);

  auto complete_boundary = std::make_shared<CompleteBoundary>();
  auto top_wall = std::make_shared<TopWall>();
  auto bottom_wall = std::make_shared<BottomWall>();
  auto top_bottom = std::make_shared<TopBottom>();
  auto left_right = std::make_shared<LeftRight>();
  auto left = std::make_shared<Left>();
  auto right = std::make_shared<Right>();

  auto rho_high = std::make_shared<Constant>(rho1);
  auto rho_low = std::make_shared<Constant>(rho2);
  auto T_cold = std::make_shared<Constant>(t_cold);
  auto T_warm = std::make_shared<Constant>(t_warm);

  DirichletBC noslip(V, u_in, complete_boundary);
  DirichletBC top_density(M, rho_high, top_wall);
  DirichletBC bottom_density(M, rho_low, bottom_wall);
  DirichletBC warm(M, T_warm, left);
  DirichletBC cold(M, T_cold, right);

  auto bcu = std::make_shared<std::vector<DirichletBC *>>();
  bcu->push_back(&noslip);

  auto bcp = std::make_shared<std::vector<DirichletBC *>>();
  auto bcrho = std::make_shared<std::vector<DirichletBC *>>();
  (void)top_density;
  (void)bottom_density;

  auto bcT = std::make_shared<std::vector<DirichletBC *>>();
  bcT->push_back(&warm);
  bcT->push_back(&cold);

  auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
  auto slip_bc_marker = std::make_shared<SlipBC_marker>();
  auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

  set_log_active(true);
  tic();
  fem_varden_flow::VD_NSESolver ns(mesh, *u_in, rho_in, zero, T_in, f, f_man, f_T, bcu,
                           bcp, bcrho, bcT, slip_bcs, slip_bc_marker,
                           slip_bc_marker_fn, V, Q, M, solver_parameters, t,
                           dt);
  info("Solver setup took %lf seconds", toc());
  set_log_active(false);

  if (argc < 2 || argv[1] == nullptr) {
    ns.adaptive_solve();
  } else {
    std::string const run_str = argv[1];
    if (argc < 3) {
      error("Run './demo checkpoint N' or './demo statistics N'");
    }
    int const n_retreive = atoi(argv[2]);

    if (run_str == "checkpoint") {
      ns.adaptive_solve(n_retreive);
    } else if (run_str == "statitics") {
      // fall through
    } else {
      error("Run './demo checkpoint N' or './demo statistics N'");
    }
  }

  MPI_Finalize();
  return 0;
}
