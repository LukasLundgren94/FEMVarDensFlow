// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>

/// varden
#include "varden/solver/ufl/VD_Mixed_2D_P1P1.h"

#include "Norms.h"
#include <ConfigurableParameters.h>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSESolver.h>
#include <varden/solver/VD_NSETracerEquation.h>

/// core
#include <SlipBC.h>

#include "Well.h"

using namespace dolfin;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  fem_varden_flow::ConfigurableParameters config_params("parameters.json");
  Parameters solver_parameters =
      config_params.to_dolfin_parameters("solver_parameters");
  skew_rho = solver_parameters["skew_rho_factor"];

  bool unstructured_mesh = true;
  bool outflow_bc = false;
  bool use_slip = false;
  bool linearProfile = false;
  bool inflow_bc = false;
  bool periodic = false;
  (void)linearProfile;

  parameters["std_out_all_processes"] = false;
  parameters["allow_extrapolation"] = true;
  parameters["refinement_algorithm"] = "plaza";
  parameters["reorder_dofs_serial"] = false;

  const int n_points = solver_parameters["n_points"];
  const std::size_t N = static_cast<std::size_t>(1) << n_points;

  std::cout << "V:" << static_cast<int>(solver_parameters["PolDegree_V_M"])
            << std::endl;
  std::cout << "Q:" << static_cast<int>(solver_parameters["PolDegree_Q"])
            << std::endl;

  auto mesh = std::make_shared<Mesh>();
  if (unstructured_mesh) {
    std::string filename = "square" + std::to_string(n_points) + ".xdmf";
    XDMFFile mesh_file(filename);
    mesh_file.read(*mesh);
  } else {
    Point p0(xmin, ymin);
    Point p1(xmax, ymax);
    mesh = std::make_shared<Mesh>(RectangleMesh::create(
        {{p0, p1}}, {{N, N}}, CellType::Type::triangle, "left"));
  }
  initialize_setup(mesh);

  double t = 0.0;
  double dt = 0.0;
  time_t tstart = time(nullptr);

  auto T_bar = std::make_shared<double>(10.5);
  auto rho_in = std::make_shared<IndataDensity>(t);
  auto S_in = std::make_shared<IndataSalt>(t);
  auto T_in = std::make_shared<IndataTemperature>(t);
  auto p_in = std::make_shared<IndataPressure>(t, T_bar);

  std::shared_ptr<FunctionSpace> V;
  std::shared_ptr<FunctionSpace> Q;
  std::shared_ptr<FunctionSpace> M;
  std::shared_ptr<FunctionSpace> V1;

  std::shared_ptr<Constant> zeros;
  std::shared_ptr<Constant> u_in;
  std::shared_ptr<Constant> f;
  std::shared_ptr<Constant> f_u_source;

  std::shared_ptr<PeriodicBoundary> pb;
  std::shared_ptr<const SubDomain> periodic_boundary;
  if (periodic) {
    pb = std::make_shared<PeriodicBoundary>();
    periodic_boundary = std::static_pointer_cast<const SubDomain>(pb);
  }

  if (periodic_boundary) {
    V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(
        mesh, periodic_boundary, solver_parameters);
    Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(
        mesh, periodic_boundary, solver_parameters);
    M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(
        mesh, periodic_boundary, solver_parameters);
  } else {
    V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh,
                                                          solver_parameters);
    Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(mesh,
                                                            solver_parameters);
    M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(mesh,
                                                           solver_parameters);
  }

  if (periodic && pb) {
    V1 = std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh, pb);
  } else {
    V1 = std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh);
  }
  zeros = std::make_shared<Constant>(0.0, 0.0);
  u_in = std::make_shared<Constant>(0.0, 0.0);
  f = std::make_shared<Constant>(0.0, 0.0);

  auto zero = std::make_shared<Constant>(0.0);
  auto u_inflow = std::make_shared<Constant>(1.0, 0.0);

  auto f_T = std::make_shared<Constant>(0.0, 1.0);
  auto f_S = std::make_shared<Constant>(0.0, 0.0);
  f_u_source = std::make_shared<Constant>(0.0, 0.0);

  auto sub_domains = std::make_shared<MeshFunction<std::size_t>>(
      mesh, mesh->topology().dim() - 1, 0);
  auto complete_boundary = std::make_shared<CompleteBoundary>();
  auto top_bottom = std::make_shared<TopBottom>();
  auto left = std::make_shared<Left>();
  auto right = std::make_shared<Right>();
  auto left_right = std::make_shared<LeftRight>();

  complete_boundary->mark(*sub_domains, 1);

  auto bc_inflow_T = std::make_shared<DirichletBC>(M, T_in, left);
  auto bc_outflow_T = std::make_shared<DirichletBC>(M, T_in, right);
  auto bc_inflow_S = std::make_shared<DirichletBC>(M, S_in, left);
  auto bc_outflow_S = std::make_shared<DirichletBC>(M, S_in, right);

  auto noslip_tb = std::make_shared<DirichletBC>(V, u_in, top_bottom);
  auto noslip_left = std::make_shared<DirichletBC>(V, u_in, left);
  auto noslip_right = std::make_shared<DirichletBC>(V, u_in, right);
  auto noslip_lr = std::make_shared<DirichletBC>(V, u_in, left_right);

  auto slip_tb = std::make_shared<DirichletBC>(V->sub(1), zero, top_bottom);
  auto slip_lr = std::make_shared<DirichletBC>(V->sub(0), zero, left_right);
  auto slip_left_comp = std::make_shared<DirichletBC>(V->sub(0), zero, left);
  auto inflow_velocity_bc = std::make_shared<DirichletBC>(V, u_inflow, left);

  auto bcu = std::make_shared<std::vector<DirichletBC *>>();
  if (!periodic) {
    if (use_slip) {
      bcu->push_back(slip_tb.get());
      if (!outflow_bc && !inflow_bc) {
        bcu->push_back(slip_lr.get());
      } else if (inflow_bc) {
        bcu->push_back(inflow_velocity_bc.get());
      } else {
        bcu->push_back(slip_left_comp.get());
      }
    } else {
      bcu->push_back(noslip_tb.get());
      if (!outflow_bc && !inflow_bc) {
        bcu->push_back(noslip_lr.get());
      } else if (inflow_bc) {
        bcu->push_back(inflow_velocity_bc.get());
      } else {
        bcu->push_back(noslip_left.get());
      }
      if (!outflow_bc) {
        bcu->push_back(noslip_right.get());
      }
    }
  }

  auto bcp = std::make_shared<std::vector<DirichletBC *>>();
  auto bcrho = std::make_shared<std::vector<DirichletBC *>>();
  auto bcT = std::make_shared<std::vector<DirichletBC *>>();
  auto bcS = std::make_shared<std::vector<DirichletBC *>>();
  if (inflow_bc) {
    bcT->push_back(bc_inflow_T.get());
    bcS->push_back(bc_inflow_S.get());
  }
  if (outflow_bc) {
    bcT->push_back(bc_outflow_T.get());
    bcS->push_back(bc_outflow_S.get());
  }

  fem_varden_flow::SlipBC slip(V, V1, complete_boundary);
  auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
  if (use_slip) {
    slip_bcs->push_back(&slip);
  }

  auto slip_bc_marker = std::make_shared<SlipBC_marker>();
  auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

  auto f_source_T = std::make_shared<Constant>(0.0);
  auto f_source_S = std::make_shared<Constant>(0.0);
  auto f_reaction_T = std::make_shared<Constant>(0.0);
  auto f_reaction_S = std::make_shared<Constant>(0.0);

  auto u_initial = inflow_bc ? u_inflow : zeros;

  fem_varden_flow::VD_NSESolver ns(mesh, *u_initial, rho_in, p_in, T_in, S_in, f,
                           f_u_source, f_T, f_S, f_source_T, f_reaction_T,
                           f_source_S, f_reaction_S, bcu, bcp, bcrho, bcT, bcS,
                           slip_bcs, slip_bc_marker, slip_bc_marker_fn, V, Q, M,
                           solver_parameters, sub_domains, t, dt);

  T_bar = ns.get_T_bar();
  p_in->T_bar_ = T_bar;

  ns.solve();

  auto Fu = ns.get_Fu();
  auto FP = ns.get_FP();
  auto FS = ns.get_FS();
  auto FT = ns.get_FT();

  fem_varden_flow::Norms::compute_absolute_error(*FS, *S_in, "Salinity");
  fem_varden_flow::Norms::compute_absolute_error(*FT, *T_in, "Temperature");
  fem_varden_flow::Norms::compute_absolute_error(*Fu, *u_initial, "Velocity");
  fem_varden_flow::Norms::compute_absolute_error(*FP, *p_in, "Pressure");

  time_t tend = time(nullptr);
  info("Solve loop took %f seconds", difftime(tend, tstart));
  MPI_Finalize();
  return 0;
}
