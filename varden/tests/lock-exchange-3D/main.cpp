// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2022-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <cmath>
#include <ctime>
#include <fstream>
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
#include <dolfin/la/solve.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/generation/BoxMesh.h>

/// varden
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSETracerEquation.h>
#include <varden/solver/VD_NSESolver.h>

/// core
#include <SlipBC.h>

#include "Lock-exchange.h"

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  {
    fem_varden_flow::ConfigurableParameters config_params("parameters.json");
    Parameters solver_parameters =
        config_params.to_dolfin_parameters("solver_parameters");

    parameters["std_out_all_processes"] = false;
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza";

    const int pol_degree_V_M = solver_parameters["PolDegree_V_M"];
    const int pol_degree_Q = solver_parameters["PolDegree_Q"];
    const int Ny = solver_parameters["n_points"];
    const int Nz = Ny;
    const int Nx = 9 * Ny;

    std::cout << "V:" << pol_degree_V_M << std::endl;
    std::cout << "Q:" << pol_degree_Q << std::endl;

    Point p0(xmin, ymin, zmin);
    Point p1(xmax, ymax, zmax);
    auto mesh = std::make_shared<Mesh>(BoxMesh::create(
        {{p0, p1}},
        {{static_cast<std::size_t>(Nx), static_cast<std::size_t>(Ny),
          static_cast<std::size_t>(Nz)}},
        CellType::Type::tetrahedron));

    double t = 0.0;
    double dt = 0.0;
    time_t tstart = time(nullptr);

    auto rho_in = std::make_shared<IndataDensity>(t);
    auto u_in = std::make_shared<IndataVelocity3D>();

    auto zero = std::make_shared<Constant>(0.0);
    auto f = std::make_shared<Constant>(0.0, -std::sqrt(g), 0.0);
    auto f_man = std::make_shared<Constant>(0.0, 0.0, 0.0);

    auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(
        mesh, solver_parameters);
    auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
        mesh, solver_parameters);
    auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
        mesh, solver_parameters);

    auto top_bottom = std::make_shared<TopBottom>();
    auto left_right = std::make_shared<LeftRight>();
    DirichletBC slip_manual_tb(V->sub(1), zero, top_bottom);
    DirichletBC slip_manual_lr(V->sub(0), zero, left_right);

    auto bcu = std::make_shared<std::vector<DirichletBC *>>();
    bcu->push_back(&slip_manual_tb);
    bcu->push_back(&slip_manual_lr);

    auto bcp = std::make_shared<std::vector<DirichletBC *>>();
    auto bcrho = std::make_shared<std::vector<DirichletBC *>>();

    auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
    auto slip_bc_marker = std::make_shared<SlipBC_marker>();
    auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

    double const Re = solver_parameters["nu"];
    double const nu =
        rho1 * H * std::sqrt((rho1 - rho2) * g) / Re; // convert Re to nu
    solver_parameters["nu"] = nu;

    fem_varden_flow::VD_NSESolver ns(mesh, *u_in, rho_in, zero, f, f_man, bcu, bcp,
                             bcrho, slip_bcs, slip_bc_marker, slip_bc_marker_fn,
                             V, Q, M, solver_parameters, t, dt);

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
      } else {
        error("Run './demo checkpoint N' or './demo statistics N'");
      }
    }

    time_t tend = time(nullptr);
    info("It took %f sec", difftime(tend, tstart));
  }
  MPI_Finalize();
  return 0;
}
