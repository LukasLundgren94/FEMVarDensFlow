// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
// Copyright (c) 2025 Jonathan Wiskandt
// Licensed under the MIT License – see the LICENSE file for details

#include <ctime>

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubMesh.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>

/// varden
#include "varden/solver/ufl/VD_Mixed_2D_P1P1.h"
#include "varden/solver/ufl/VD_Mixed_3D_P1P1.h"

#include <ConfigurableParameters.h>
#include <utility>
#include <varden/solver/VD_NSEMomEquation.h>
#include <varden/solver/VD_NSEPressEquation.h>
#include <varden/solver/VD_NSESolver.h>
#include <varden/solver/VD_NSETracerEquation.h>

/// core
#include <SlipBC.h>

#include "Ocean.h"
#include <mpi.h>

using namespace dolfin;

namespace
{

  template <typename... Args>
  void log_info_once(char const *msg, Args &&...args)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0)
      return;

    set_log_active(true);
    info(msg, std::forward<Args>(args)...);
    set_log_active(false);
  }

  template <typename Setter>
  void apply_parameter_if_present(Parameters const &params, char const *key,
                                  Setter setter, char const *missing_message)
  {
    try
    {
      setter(params[key]);
    }
    catch (...)
    {
      if (missing_message)
      {
        log_info_once("%s", missing_message);
      }
    }
  }

} // namespace

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  bool noslip = true;
  bool noslip_outflow = false;
  bool outflow_dirichlet = false;
  int checkpoint_restart_index = -1;

  // Print log messages only from the root process in parallel
  // the followings are global Dolfin parameters
  parameters["std_out_all_processes"] = false;
  parameters["allow_extrapolation"] = true;
  // parameters["ghost_mode"] = "shared_facet";
  // parameters["ghost_mode"] = "shared_vertex";
  // parameters["refinement_algorithm"] = "plaza";
  // fem_varden_flow::ConfigurableParameters config_params;

  fem_varden_flow::ConfigurableParameters config_params("parameters.json");

  auto test = config_params.values();
  std::string test_string = boost::get<std::string>(test["model_setup.mesh"]);
  x_r = boost::get<double>(test["restoring_region_width"]);
  double skew_factor = boost::get<double>(test["skew_rho_factor"]);
  int remove_rho_average = boost::get<int>(test["remove_rho_average"]);
  log_info_once("Mesh from config: %s", test_string.c_str());
  Parameters solver_parameters =
      config_params.to_dolfin_parameters("solver_parameters");

  g = solver_parameters["gravitational_acceleration"];
  S_0 = solver_parameters["S_0"];
  T_0 = solver_parameters["T_0"];
  alpha_T = solver_parameters["alpha_T"];
  beta_S = solver_parameters["beta_S"];
  double const restoring_region_width =
      solver_parameters["restoring_region_width"];

  std::string const file_name_mesh = solver_parameters["mesh"];

  checkpoint_restart_index = solver_parameters["checkpoint_restart_index"];

  int melt_with_virtual_flux = solver_parameters["melt_with_virtual_flux"];

  if (restoring_region_width > 0.0)
  {
    outflow_dirichlet = true;
  }

  parameters["refinement_algorithm"] = "plaza";
  parameters["reorder_dofs_serial"] = false;

  time_t tstart, tend;

  int const nRefinements = solver_parameters["n_points"];

  auto mesh = std::make_shared<Mesh>(MPI_COMM_WORLD);
  XDMFFile mesh_file(file_name_mesh);

  mesh_file.read(*mesh);

  tic();
  for (int i = 0; i < nRefinements; i++)
  {
    log_info_once("Mesh refinement %d", i);
    *mesh = refine(*mesh);
  }
  log_info_once("Mesh generation took %lf seconds", toc());
  // ********************************************************

  // Make sure ymax = 0
  // Assume that mesh is oriented so that y is pointing up
  fix_max_min(mesh);

  Point point2(0.0, -ymax);
  mesh->translate(point2);
  fix_max_min(mesh);

  log_info_once("xmax = %lf", xmax);
  log_info_once("ymax = %lf", ymax);
  log_info_once("xmin = %lf", xmin);
  log_info_once("ymin = %lf", ymin);

  // local_refinement(*mesh);
  // some operations to fix coordinates in Ryder geometry
  // mesh->rotate(90,0);
  // mesh->rotate(-116,1);
  // Point point1(-xmin, -ymin,0.0);
  // mesh->translate(point1);
  // mesh->smooth(5);
  // XDMFFile save_file("test_mesh.xdmf");
  // save_file.write(*mesh,XDMFFile::Encoding::HDF5);
  // return 0;

  double t;
  auto rho_in = std::make_shared<IndataDensity>(t);
  auto T_in = std::make_shared<IndataTemperature>(t);
  auto S_in = std::make_shared<IndataSalt>(t);
  std::shared_ptr<double> S_bar = std::make_shared<double>(10.5);
  std::shared_ptr<double> T_bar = std::make_shared<double>(10.5);
  auto p_in = std::make_shared<IndataPressure>(t, T_bar, S_bar);

  // Create function spaces
  int const D = mesh->topology().dim();

  std::shared_ptr<FunctionSpace> V, Q, M, V1;
  std::shared_ptr<Constant> u_in, f, f_u_source, f_T, f_S;

  std::shared_ptr<Constant> zeros;

  auto mesh_fine = std::make_shared<Mesh>(*mesh);
  refine(*mesh_fine, *mesh, false);
  refine(*mesh_fine, *mesh_fine, false);

  V = fem_varden_flow::VD_NSEMomEquation::create_velocity_space(mesh,
                                                                solver_parameters);
  Q = fem_varden_flow::VD_NSEPressEquation::create_pressure_space(mesh,
                                                                  solver_parameters);
  M = fem_varden_flow::VD_NSETracerEquation::create_tracer_space(mesh,
                                                                 solver_parameters);

  if (D == 2)
  {
    V1 = std::make_shared<VD_Mixed_2D_P1P1::CoefficientSpace_u_n>(mesh_fine);
    zeros = std::make_shared<Constant>(0, 0);
    u_in = std::make_shared<Constant>(0.0, 0.0);
    f = std::make_shared<Constant>(0.0, 0.0);
    // Buouancy forces
    f_T = std::make_shared<Constant>(0.0, g * alpha_T);
    f_S = std::make_shared<Constant>(0.0, -g * beta_S);
    f_u_source =
        std::make_shared<Constant>(0.0, g * (-alpha_T * T_0 + beta_S * S_0));
  }
  else if (D == 3)
  {
    V1 = std::make_shared<VD_Mixed_3D_P1P1::CoefficientSpace_u_n>(mesh_fine);
    zeros = std::make_shared<Constant>(0, 0, 0);
    u_in = std::make_shared<Constant>(0.0, 0.0, 0.0);
    f = std::make_shared<Constant>(0.0, 0.0, 0.0);
    // Buouancy forces
    f_T = std::make_shared<Constant>(0.0, g * alpha_T, 0.0);
    f_S = std::make_shared<Constant>(0.0, -g * beta_S, 0.0);
    f_u_source = std::make_shared<Constant>(
        0.0, g * (-alpha_T * T_0 + beta_S * S_0), 0.0);
  }

  auto zero = std::make_shared<Constant>(0);

  auto complete_boundary = std::make_shared<CompleteBoundary>();

  // create mesh function:
  auto top_wall = std::make_shared<TopWall>();
  auto left = std::make_shared<Left>();
  auto right = std::make_shared<Right>();
  auto outflow_corners_top = std::make_shared<CornersOutflowTop>();
  auto outflow_corners_bottom = std::make_shared<CornersOutflowBottom>();
  auto cross_section = std::make_shared<CrossSection>();

  auto sub_domains = std::make_shared<MeshFunction<std::size_t>>(
      mesh, mesh->topology().dim() - 1, 0);

  try
  {
    std::string const file_name_surface_mesh =
        solver_parameters["surface_mesh"];
    log_info_once("Marking subdomains from file");
    XDMFFile mvc_xdmf(file_name_surface_mesh);
    MeshValueCollection<std::size_t> mvc(mesh);

    mvc_xdmf.read(mvc,
                  "name_to_read"); // read Attribute "name" from the right Grid

    MeshFunction<std::size_t> cell_markers(mesh, mvc);
    *sub_domains = cell_markers;
  }
  catch (...)
  {
    log_info_once("Surface_mesh not found in parameters, attempting to mark "
                  "subdomains manually.");

    complete_boundary->mark(
        *sub_domains,
        1);                                          // marks all boundary facets as 1. This represents ice shelf.
    cross_section->mark(*sub_domains, 6);            // important to mark first???
    int direction = -1;                              // corresponds to down in christians code
    mark_facet_direction(sub_domains, direction, 2); // bed
    left->mark(*sub_domains, 3);                     // left inflow
    top_wall->mark(*sub_domains, 5);                 // top
    right->mark(*sub_domains, 4);                    // right outflow
    outflow_corners_top->mark(*sub_domains, 7);
    outflow_corners_bottom->mark(*sub_domains, 8);

    if (D == 3)
    {
      auto z_wall = std::make_shared<ZWall>();
      z_wall->mark(*sub_domains, 2); // z min and max
    }
  }

  File("results/pvd_files/subdomains.pvd") << *sub_domains;
  DirichletBC bc_inflow_T(M, T_in, sub_domains, 3);  // left
  DirichletBC bc_outflow_T(M, T_in, sub_domains, 4); // right
  DirichletBC bc_inflow_S(M, S_in, sub_domains, 3);  // left
  DirichletBC bc_outflow_S(M, S_in, sub_domains, 4); // right
  // DirichletBC outflow_slip(V->sub(1), zero, sub_domains, 5);
  DirichletBC outflow_slip(V->sub(1), zero, top_wall);
  DirichletBC outflow_corners_slip_top(V->sub(1), zero, outflow_corners_top);
  DirichletBC outflow_corners_slip_bottom(V->sub(1), zero,
                                          outflow_corners_bottom);
  DirichletBC outflow_corners_noslip_top(V, zeros, outflow_corners_top);
  DirichletBC outflow_corners_noslip_bottom(V, zeros, outflow_corners_bottom);
  // DirichletBC outflow_corners_noslip(V, u_in, outflow_corners);

  DirichletBC noslip_ice(V, u_in, sub_domains, 1);
  DirichletBC noslip_bed(V, u_in, sub_domains, 2);
  DirichletBC noslip_left(V, u_in, sub_domains, 3);
  DirichletBC noslip_right(V, u_in, sub_domains, 4);
  DirichletBC noslip_top(V, u_in, sub_domains, 5);

  DirichletBC outflow_p(Q, zero, sub_domains, 4); // ?????
  DirichletBC top_p(Q, p_in, top_wall);
  DirichletBC top_corner_p(Q, p_in, outflow_corners_top);
  DirichletBC bottom_corner_p(Q, p_in, outflow_corners_bottom);
  DirichletBC p_everywhere(Q, p_in, complete_boundary);
  DirichletBC ice_shelf_p(Q, p_in, sub_domains, 1);
  DirichletBC bottom_p(Q, p_in, sub_domains, 2);

  std::shared_ptr<std::vector<DirichletBC *>> bcu =
      std::make_shared<std::vector<DirichletBC *>>();
  // auto &u_dirichlet = bcu.front();
  if (noslip)
  {
    bcu->push_back(&noslip_bed);
    if (melt_with_virtual_flux == 1)
    {
      bcu->push_back(&noslip_ice);
    }
    bcu->push_back(&noslip_left);
    bcu->push_back(&outflow_corners_noslip_top);
    bcu->push_back(&outflow_corners_noslip_bottom);
  }
  else
  {
    bcu->push_back(&outflow_corners_slip_top);
    bcu->push_back(&outflow_corners_slip_bottom);
  }
  if (noslip_outflow)
  {
    // u_dirichlet.push_back(&noslip_right);
    bcu->push_back(&noslip_right);
  }

  auto bcp = std::make_shared<std::vector<DirichletBC *>>();

  bcp->push_back(&top_p);
  bcp->push_back(&top_corner_p);
  bcp->push_back(&ice_shelf_p);
  // bcp->push_back(&p_everywhere);

  std::shared_ptr<std::vector<DirichletBC *>> bcrho =
      std::make_shared<std::vector<DirichletBC *>>();

  // Collect boundary condition of temperature: Should it be const or not?
  auto bcT = std::make_shared<std::vector<DirichletBC *>>();
  // bcT->push_back(&bc_inflow_T); // add later when grounding line is
  // included...
  if (outflow_dirichlet)
  {
    bcT->push_back(&bc_outflow_T);
  }

  auto bcS = std::make_shared<std::vector<DirichletBC *>>();
  if (outflow_dirichlet)
  {
    bcS->push_back(&bc_outflow_S);
  }

  auto slip_bcs = std::make_shared<std::vector<fem_varden_flow::SlipBC *>>();
  fem_varden_flow::SlipBC slip(V, V1, complete_boundary);
  fem_varden_flow::SlipBC slip_ice(V, V1, sub_domains, 1);
  fem_varden_flow::SlipBC slip_bed(V, V1, sub_domains, 2);
  fem_varden_flow::SlipBC slip_left(V, V1, sub_domains, 3);
  fem_varden_flow::SlipBC slip_top(V, V1, sub_domains, 5);
  if (!noslip)
  {
    slip_bcs->insert(slip_bcs->end(), {&slip_ice, &slip_bed, &slip_left});
  }
  // now we use slip on the atmosphere boundary
  slip_bcs->push_back(&slip_top);

  // I should probably mark my Neumann BC as well
  auto slip_bc_marker = std::make_shared<SlipBC_marker>();
  auto slip_bc_marker_fn = std::make_shared<SlipBC_marker_fn>();

  dolfin::list_linear_solver_methods();
  dolfin::list_krylov_solver_preconditioners();

  auto f_source_T = std::make_shared<Forcing_T>(t);
  auto f_source_S = std::make_shared<Forcing_S>(t);
  auto f_reaction_T = std::make_shared<Reaction_T>(t);
  auto f_reaction_S = std::make_shared<Reaction_S>(t);

  tic();
  fem_varden_flow::VD_NSESolver ns(mesh, *u_in, rho_in, p_in, T_in, S_in, f, f_u_source,
                                   f_T, f_S, f_source_T, f_reaction_T, f_source_S,
                                   f_reaction_S, bcu, bcp, bcrho, bcT, bcS, slip_bcs,
                                   slip_bc_marker, slip_bc_marker_fn, V, Q, M,
                                   solver_parameters, sub_domains, t, 0);
  log_info_once("Solver setup took %lf seconds", toc());

  S_bar = ns.get_S_bar();
  T_bar = ns.get_T_bar();
  p_in->S_bar_ = S_bar;
  p_in->T_bar_ = T_bar;

  tstart = time(0);
  if (checkpoint_restart_index <= 0)
  {
    ns.adaptive_solve();
  }
  else
  {
    ns.adaptive_solve(checkpoint_restart_index);
  }
  tend = time(0);
  log_info_once("Solve loop took %lf seconds", difftime(tend, tstart));

  // useful when debugging pressure shift
  // File p_file("p_analytic.pvd");
  // Function p_temp(Q);
  // p_temp.interpolate(*p_in);
  // p_file << p_temp;
  MPI_Finalize();
  return 0;
}
