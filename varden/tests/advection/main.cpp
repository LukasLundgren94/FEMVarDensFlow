// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2021-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <cmath>
#include <ctime>
#include <mpi.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/la/solve.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>

/// varden
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSETracerEquation.h>
#include <varden/solver/VD_NSESolver.h>

/// core
#include <SlipBC.h>

#include "Advection.h"
#include "Norms.h"

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
    fem_varden_flow::ConfigurableParameters config_params("parameters.json");
    Parameters solver_parameters = config_params.to_dolfin_parameters("solver_parameters");

    std::cout << "Start:" << std::endl;
    parameters["std_out_all_processes"] = false;
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza";

    std::vector<int> nPoints = {4,   8,    16,    32,    64,
                                128, 256,  512,   1024,  2048};

    xmin = -1.0;
    ymin = -1.0;
    ymax = 1.0;
    xmax = 1.0;

    const int pol_degree_V_M = solver_parameters["PolDegree_V_M"];
    const int pol_degree_Q = solver_parameters["PolDegree_Q"];
    const int start_ind = solver_parameters["n_points"];


    for (int i = start_ind; i < static_cast<int>(nPoints.size()); ++i) {
      time_t tstart = time(nullptr);
      double t = 0.0;
      double dt = 0.0;

      auto mesh = std::make_shared<Mesh>();
      XDMFFile mesh_file("square" + std::to_string(i) + ".xdmf");
      mesh_file.read(*mesh);

      auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh,
                                                                 solver_parameters);
      auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(mesh,
                                                                   solver_parameters);
      auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(mesh,
                                                                  solver_parameters);

      auto f = std::make_shared<Constant>(0.0, 0.0);
      auto f_man = std::make_shared<Constant>(0.0, 0.0);
      auto u_analytic = std::make_shared<AnalyticVelocity>(t);
      auto p_analytic = std::make_shared<AnalyticPressure>(t);
      auto rho_analytic = std::make_shared<AnalyticDensity>(t);
      // auto rho_analytic = std::make_shared<AnalyticDensityCylinder>(t);
      auto zero = std::make_shared<Constant>(0.0);

      auto complete_boundary = std::make_shared<CompleteBoundary>();

      DirichletBC velocityBC(V, u_analytic, complete_boundary);
      DirichletBC densityBC(M, rho_analytic, complete_boundary);
      DirichletBC pressureBC(Q, p_analytic, complete_boundary);

      auto bcu = std::make_shared<std::vector<DirichletBC *>>();
      bcu->push_back(&velocityBC);

      auto bcp = std::make_shared<std::vector<DirichletBC *>>();
      auto bcrho = std::make_shared<std::vector<DirichletBC *>>();
      bcrho->push_back(&densityBC);

      auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
      auto slip_bc_marker = std::make_shared<SlipBC_marker>();
      auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

      fem_varden_flow::VD_NSESolver ns(
          mesh, *u_analytic, rho_analytic, zero, f, f_man, bcu, bcp, bcrho,
          slip_bcs, slip_bc_marker, slip_bc_marker_fn, V, Q, M, solver_parameters,
          t, dt);

      std::shared_ptr<Function> Fh = ns.get_Fh();
      u_analytic->hmin = Fh->vector()->min();

      // FIXME: add checkpointing later
      ns.adaptive_solve();
      // ns.solve();


      std::shared_ptr<Function> Fu = ns.get_Fu();
      std::shared_ptr<Function> Frho = ns.get_Frho();
      std::shared_ptr<Function> FP = ns.get_FP();

      int rank = 0;
      MPI_Comm_rank(mesh->mpi_comm(), &rank);

      if (rank == 0) {
        // keep hook for optional debug output.
      }
      fem_varden_flow::Norms::compute_error(*FP, *p_analytic, "Pressure");
      fem_varden_flow::Norms::compute_error(*Frho, *rho_analytic, "Density");
      fem_varden_flow::Norms::compute_error(*Fu, *u_analytic, "Velocity");

      time_t tend = time(nullptr);
      info("It took %f sec", difftime(tend, tstart));
    }
  MPI_Finalize();
  return 0;

}
