// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2022-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
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
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>

/// varden
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSETracerEquation.h>
#include <varden/solver/VD_NSESolver.h>

#include "Gresho.h"
#include "Norms.h"

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

    std::cout << "Start:" << std::endl;

    const int pol_degree_V_M = solver_parameters["PolDegree_V_M"];
    const int pol_degree_Q = solver_parameters["PolDegree_Q"];
    const int n_points = solver_parameters["n_points"];

    std::cout << "V:" << pol_degree_V_M << std::endl;
    std::cout << "Q:" << pol_degree_Q << std::endl;

    const std::size_t N = static_cast<std::size_t>(1) << n_points;
    Point p0(xmin, ymin);
    Point p1(xmax, ymax);

    auto mesh = std::make_shared<Mesh>(RectangleMesh::create(
        {{p0, p1}}, {{N, N}}, CellType::Type::triangle, "crossed"));

    double t = 0.0;
    double dt = 0.0;
    time_t tstart = time(nullptr);

    auto u_analytic = std::make_shared<AnalyticVelocity>(t);
    auto p_analytic = std::make_shared<AnalyticPressure>(t);
    auto rho_analytic = std::make_shared<AnalyticDensity>(t);

    std::shared_ptr<FunctionSpace> V;
    std::shared_ptr<FunctionSpace> Q;
    std::shared_ptr<FunctionSpace> M;
    std::shared_ptr<Constant> f;
    std::shared_ptr<const Expression> f_man;

    if (mesh->topology().dim() == 2) {
      auto pb = std::make_shared<PeriodicBoundaryXY1>();
      V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh, pb,
                                                            solver_parameters);
      Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(mesh, pb,
                                                              solver_parameters);
      M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(mesh, pb,
                                                             solver_parameters);
      f = std::make_shared<Constant>(0.0, 0.0);
      f_man = std::make_shared<ForcingFunction>(t);
    } else {
      V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(
          mesh, solver_parameters);
      Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
          mesh, solver_parameters);
      M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
          mesh, solver_parameters);
      f = std::make_shared<Constant>(0.0, 0.0, 0.0);
      f_man = std::make_shared<ForcingFunction3D>(t);
    }

    auto bcu = std::make_shared<std::vector<DirichletBC *>>();
    auto bcp = std::make_shared<std::vector<DirichletBC *>>();
    auto bcrho = std::make_shared<std::vector<DirichletBC *>>();

    list_linear_solver_methods();
    list_linear_algebra_backends();

    fem_varden_flow::VD_NSESolver ns(mesh, *u_analytic, rho_analytic, p_analytic, f,
                             f_man, bcu, bcp, bcrho, V, Q, M,
                             solver_parameters, t, dt);

    if (argc < 2 || argv[1] == nullptr) {
      ns.solve();
    } else {
      std::string const run_str = argv[1];
      if (argc < 3) {
        error("Run './demo checkpoint N' or './demo statistics N'");
      }
      int const n_retreive = atoi(argv[2]);

      if (run_str == "checkpoint") {
        ns.solve(n_retreive);
      } else if (run_str == "statitics") {
        // fall through
      } else {
        error("Run './demo checkpoint N' or './demo statistics N'");
      }
    }

    auto Fu = ns.get_Fu();
    auto Frho = ns.get_Frho();
    auto FP = ns.get_FP();

    int rank = 0;
    MPI_Comm_rank(mesh->mpi_comm(), &rank);

    if (rank == 0) {
      // hook retained for optional debugging output.
    }
    fem_varden_flow::Norms::compute_error(*FP, *p_analytic, "Pressure");

    if (rank == 0) {
      // hook retained for optional debugging output.
    }
    fem_varden_flow::Norms::compute_error(*Frho, *rho_analytic, "Density");

    if (rank == 0) {
      // hook retained for optional debugging output.
    }
    fem_varden_flow::Norms::compute_error(*Fu, *u_analytic, "Velocity");


    time_t tend = time(nullptr);
    info("It took %f sec", difftime(tend, tstart));
  }
  MPI_Finalize();
  return 0;
}
