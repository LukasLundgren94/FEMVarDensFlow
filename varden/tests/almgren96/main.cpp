// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2022-2025 Lukas Lundgren
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
#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/io/HDF5File.h>
#include <dolfin/la/solve.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>

/// varden
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSETracerEquation.h>
#include <varden/solver/VD_NSESolver.h>

#include "Almgren.h"
#include "FileSaver.h"
#include "Norms.h"

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  {
    fem_varden_flow::ConfigurableParameters config_params("parameters.json");
      Parameters solver_parameters = config_params.to_dolfin_parameters("solver_parameters");

    parameters["std_out_all_processes"] = false;
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza";
    parameters["dof_ordering_library"] = "SCOTCH";

    std::cout << "Start:" << std::endl;

    const int pol_degree_V_M = solver_parameters["PolDegree_V_M"];
    const int pol_degree_Q = solver_parameters["PolDegree_Q"];
    const std::size_t N = solver_parameters["n_points"];
    const int number_of_refinements = solver_parameters["number_of_refinements"];
    const int save_ref_sol = solver_parameters["save_reference_solution"];

    std::cout << "V:" << pol_degree_V_M << std::endl;
    std::cout << "Q:" << pol_degree_Q << std::endl;

    Point p0(xmin, ymin);
    Point p1(xmax, ymax);

    auto mesh = std::make_shared<Mesh>();
    auto periodic = std::make_shared<PeriodicBoundaryXY1>();

    for (int level = 0; level <= number_of_refinements; ++level) {
      std::string mesh_file_name = "mesh_reference" + std::to_string(level) + ".xdmf";

      if (save_ref_sol && level == 0) {
        mesh = std::make_shared<Mesh>(RectangleMesh::create(
            {{p0, p1}},
            {{static_cast<std::size_t>(N), static_cast<std::size_t>(N)}},
            CellType::Type::triangle, "right/left"));
      } else if (!save_ref_sol) {
        auto mesh_file = std::make_shared<HDF5File>(MPI_COMM_WORLD, mesh_file_name, "r");
        mesh_file->read(*mesh, "mesh", true);
        mesh_file->close();
      }

      std::cout << "num cells: " << mesh->num_cells() << std::endl;

      double t = 0.0;
      double dt = 0.0;
      time_t tstart = time(nullptr);

      auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh, periodic,
                                                                  solver_parameters);
      auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(mesh, periodic,
                                                                   solver_parameters);
      auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(mesh, periodic,
                                                                  solver_parameters);

      auto f = std::make_shared<Constant>(0.0, 0.0);
      auto f_man = std::make_shared<ForcingFunction>(t);
      auto u_analytic = std::make_shared<AnalyticMomentum>(t);
      auto p_analytic = std::make_shared<AnalyticPressure>(t);
      auto rho_analytic = std::make_shared<AnalyticDensity>(t);

      auto zeros = std::make_shared<Constant>(0.0, 0.0);
      auto zero = std::make_shared<Constant>(0.0);

      auto complete_boundary = std::make_shared<CompleteBoundary>();

      DirichletBC velocityBC(V, zeros, complete_boundary);
      DirichletBC densityBC(M, rho_analytic, complete_boundary);
      DirichletBC pressureBC(Q, p_analytic, complete_boundary);

      auto bcu = std::make_shared<std::vector<DirichletBC *>>();
      bcu->push_back(&velocityBC);

      auto bcp = std::make_shared<std::vector<DirichletBC *>>();
      auto bcrho = std::make_shared<std::vector<DirichletBC *>>();
      (void)densityBC;
      (void)pressureBC;

      list_linear_solver_methods();
      list_linear_algebra_backends();

      fem_varden_flow::VD_NSESolver ns(mesh, *u_analytic, *rho_analytic, *p_analytic, *f,
                               *f_man, bcu, bcp, bcrho, V, Q, M,
                               solver_parameters, t, dt);

      if (argc < 2 || argv[1] == nullptr) {
        ns.adaptive_solve();
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

      std::shared_ptr<Function> Fu = ns.get_Fu();
      std::shared_ptr<Function> Frho = ns.get_Frho();
      std::shared_ptr<Function> FP = ns.get_FP();

      if (save_ref_sol == 1) {
        auto file_saver = std::make_shared<fem_varden_flow::FileSaver>(t, solver_parameters);
        file_saver->add_checkpoint(*Fu, "u_ref");
        file_saver->add_checkpoint(*FP, "p_ref");
        file_saver->add_checkpoint(*Frho, "rho_ref");
        if (level == number_of_refinements) {
          file_saver->write_checkpoint();
        }

        auto mesh_file = std::make_shared<HDF5File>(MPI_COMM_WORLD, mesh_file_name, "w");
        mesh_file->write(*mesh, "mesh");
        mesh_file->close();
      } else {
        auto mesh_reference = std::make_shared<Mesh>();
        std::string mesh_file_name_fine =
            "mesh_reference" + std::to_string(number_of_refinements) + ".xdmf";
        HDF5File mesh_file(MPI_COMM_WORLD, mesh_file_name_fine, "r");
        mesh_file.read(*mesh_reference, "mesh", true);
        mesh_file.close();

        auto V_ref = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(
            mesh_reference, periodic, solver_parameters);
        auto Q_ref = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
            mesh_reference, periodic, solver_parameters);
        auto M_ref = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
            mesh_reference, periodic, solver_parameters);

        auto Fu_ref = std::make_shared<Function>(V_ref);
        auto Fu_fine = std::make_shared<Function>(V_ref);
        auto Frho_ref = std::make_shared<Function>(M_ref);
        auto Frho_fine = std::make_shared<Function>(M_ref);
        auto FP_ref = std::make_shared<Function>(Q_ref);
        auto FP_fine = std::make_shared<Function>(Q_ref);

        auto file_saver = std::make_shared<fem_varden_flow::FileSaver>(t, solver_parameters);
        file_saver->add_checkpoint(*Fu_ref, "u_ref");
        file_saver->add(*Fu_ref, "u_ref");
        file_saver->add_checkpoint(*FP_ref, "p_ref");
        file_saver->add_checkpoint(*Frho_ref, "rho_ref");
        file_saver->read_checkpoint(0);
        file_saver->save();

        LagrangeInterpolator interpolator;
        interpolator.interpolate(*Fu_fine, *Fu);
        interpolator.interpolate(*Frho_fine, *Frho);
        interpolator.interpolate(*FP_fine, *FP);

        fem_varden_flow::Norms::compute_error(*FP_fine, *FP_ref, "Pressure");
        fem_varden_flow::Norms::compute_error(*Frho_fine, *Frho_ref, "Density");
        fem_varden_flow::Norms::compute_error(*Fu_fine, *Fu_ref, "Velocity");
      }

      time_t tend = time(nullptr);
      info("It took %f sec", difftime(tend, tstart));

      *mesh = refine(*mesh, false);
    }
  }
  MPI_Finalize();
  return 0;

