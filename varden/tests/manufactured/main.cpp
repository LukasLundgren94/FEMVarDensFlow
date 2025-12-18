// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>
#include <sstream>
#include <string>

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/io/File.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/fem/assemble.h>


/// varden
#include "varden/common/ufl/VD_Functionals_2D_P4P3.h"
#include <varden/common/VD_Functionals.h>

#include <ConfigurableParameters.h>
#include <SlipBC.h>
#include <varden/solver/VD_NSETracerEquation.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include "Norms.h"
#include <varden/solver/VD_NSESolver.h>

#include "Manufactured.h"
#include <mpi.h>

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  {
    fem_varden_flow::ConfigurableParameters config_params("parameters.json");
      Parameters solver_parameters = config_params.to_dolfin_parameters("solver_parameters");

    // Print log messages only from the root process in parallel
    // the followings are global Dolfin parameters
    parameters["std_out_all_processes"] = false;
    parameters["allow_extrapolation"] = true;
    parameters["refinement_algorithm"] = "plaza";
    // only needed for DG
    // parameters["ghost_mode"] = "shared_vertex";
    // parameters["ghost_mode"] = "shared_facet";

    // MESHING **************************
    // Load mesh from file
    // auto mesh = std::make_shared<Mesh>(argv[5]);
    // auto mesh = std::make_shared<Mesh>("mesh.xml");
    // *mesh = refine(*mesh);
    std::cout << "Start:" << std::endl;
    // std::vector<int> nPoints = {2,4, 8, 16, 32, 64}; // structured mesh
    // std::vector<int> nPoints = {11486}; // 19,
    std::vector<int> nPoints = {
        19,    38,    52,    199,   760, 2954,
        18000, 11486, 73103, 110305}; //, 73103
  }
  MPI_Finalize();
  return 0;
; // unstructured mesh 19,38, 52,
                                    //199, 760,

  int N = nPoints.size();
  int const ind = solver_parameters["n_points"];
  for (int i = ind; i < N; ++i) {

    time_t tstart, tend;
    tstart = time(0);
    double dt = 0.0; // n_dts[i];
    Point p0(xmin, ymin);
    Point p1(xmax, ymax);
    // RectangleMesh rect(p0, p1, nPoints[i], nPoints[i], "right/left");
    // auto mesh = std::make_shared<Mesh>(rect);
    //  auto mesh = std::make_shared<Mesh>(
    //      UnitDiscMesh::create(MPI_COMM_WORLD, nPoints[i], degree, gdim));

    std::string str = std::to_string(nPoints[i]);

    std::string part1 = "circle_unstructured_";
    std::string part2 = ".xml";
    std::string meshName = part1 + str + part2;
    auto mesh = std::make_shared<Mesh>(meshName);
    // File fmesh(meshName);
    // fmesh << *mesh;

    // ********************************************************

    // Set parameter values
    double t = 0.0;

    // Create function spaces
    auto V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(
        mesh, solver_parameters);
    auto Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
        mesh, solver_parameters);
    auto M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
        mesh, solver_parameters);

    auto f = std::make_shared<Constant>(0.0, 0.0);
    auto f_man = std::make_shared<ForcingFunction>(t);
    auto u_analytic = std::make_shared<AnalyticVelocity>(t);
    auto p_analytic = std::make_shared<AnalyticPressure>(t);
    auto rho_analytic = std::make_shared<AnalyticDensity>(t);
    auto Form_Pavg = std::make_shared<VD_Functionals_2D_P4P3::Form_Pavg>(mesh);
    auto Form_Rhoavg = std::make_shared<VD_Functionals_2D_P4P3::Form_Pavg>(mesh);

    // Define values for boundary conditions
    auto zero = std::make_shared<Constant>(0);

    // Define subdomains for boundary conditions
    auto complete_boundary = std::make_shared<CompleteBoundary>();
    // auto point_boundary = std::make_shared<CirclePoint>();

    // Define boundary conditions
    DirichletBC velocityBC(V, u_analytic, complete_boundary);

    DirichletBC densityBC(M, rho_analytic, complete_boundary);

    DirichletBC pressureBC(Q, p_analytic, complete_boundary);

    auto bcu = std::make_shared<std::vector<DirichletBC*>>();
    bcu->push_back(&velocityBC);
    auto bcp = std::make_shared<std::vector<DirichletBC*>>();
    auto bcrho = std::make_shared<std::vector<DirichletBC*>>();
    list_linear_solver_methods();
    list_linear_algebra_backends();


      auto slip_bc_marker = std::make_shared<SlipBC_marker>();
  auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();
  auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();


    fem_varden_flow::VD_NSESolver ns(
        mesh, *u_analytic, rho_analytic, p_analytic, f, f_man,
        std::shared_ptr<const std::vector<DirichletBC*>>(bcu),
        std::shared_ptr<const std::vector<DirichletBC*>>(bcp),
        std::shared_ptr<const std::vector<DirichletBC*>>(bcrho), slip_bcs,
        slip_bc_marker, slip_bc_marker_fn, V, Q, M,
        solver_parameters, t, dt);

    double const nu = solver_parameters["nu"];

    /// check if one needs postprocessing: run
    /// ./demo     to solve the problem
    /// ./demo checkpoint -1  to read the checkpoint times
    /// ./demo checkpoint n   to read a certain time level and run
    /// ./demo statistics n   to read a certain time level and compute the
    /// statistics
    // std::cout << "About to solve... " << std::endl;
    double mesh_volume_ = ns.get_mesh_volume();

    Form_Rhoavg->set_coefficient("p",rho_analytic);
    double rho_avg =  1.0*dolfin::assemble(*Form_Rhoavg);


    Form_Pavg->set_coefficient("p",p_analytic);
    double p_avg =  1.0*dolfin::assemble(*Form_Pavg);

    if (argv[1] == NULL) {
      ns.adaptive_solve();
      // ns.solve();
    } else {
      std::string const run_str = argv[1];
      int const n_retreive = atoi(argv[2]);

      if (run_str == "checkpoint") {
        ns.adaptive_solve(n_retreive);
      } else if (run_str == "statitics") {
      } else {
        error("Run './demo checkpoint N' or './demo statistics N'");
      }
    }

    std::shared_ptr<Function> Fu = ns.get_Fu();
    std::shared_ptr<Function> Frho = ns.get_Frho();
    std::shared_ptr<Function> FP = ns.get_FP();
    int rank;

    //  fem_varden_flow::Norms error_norms(mesh);
    MPI_Comm_rank(mesh->mpi_comm(), &rank);

    fem_varden_flow::Norms::compute_error(*Frho, *rho_analytic, "Density");

    fem_varden_flow::Norms::compute_error(*Fu, *u_analytic, "Velocity");

int const time_order = solver_parameters["time_order"];


    if (time_order == 2) {
      double const dt = ns.get_final_dt(); 
      double t_pressure = p_analytic->t;
      t_pressure  -= 0.5*dt;
      p_analytic->t = t_pressure;
    }
        Form_Rhoavg->set_coefficient("p",rho_analytic);
    rho_avg =  1.0*dolfin::assemble(*Form_Rhoavg);

    p_analytic->rho_bar = rho_avg/mesh_volume_;

    Form_Pavg->set_coefficient("p",p_analytic);
    p_avg =  1.0*dolfin::assemble(*Form_Pavg);

    p_analytic->normalization_value = p_avg/mesh_volume_;


    fem_varden_flow::Norms::compute_error(*FP, *p_analytic, "Pressure");

    auto Frho_error = std::make_shared<Function>(M);
    Frho_error->interpolate(*rho_analytic);
    Frho_error->vector()->axpy(-1.0, *Frho->vector());

    File file("error_rho.pvd");
    file << *Frho_error;


    tend = time(0);
    info("It took %f sec", difftime(tend, tstart));
  }
  return 0;
}
