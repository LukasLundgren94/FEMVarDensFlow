// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include "ConfigurableParameters.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <boost/property_tree/json_parser.hpp>

using dolfin::error;
using dolfin::info;

namespace
{
using fem_varden_flow::ConfigurableParameters;
using ParameterType = ConfigurableParameters::ParameterType;
using ParameterValue = ConfigurableParameters::ParameterValue;
using ParameterSpec = ConfigurableParameters::ParameterSpec;

std::string trim_copy(const std::string& value)
{
  auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
  auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
  if (begin >= end)
    return std::string();

  return std::string(begin, end);
}

std::string to_lower_copy(const std::string& value)
{
  std::string result(value.size(), '\0');
  std::transform(value.begin(), value.end(), result.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

bool try_parse_bool(const std::string& input, bool& output)
{
  const std::string lowered = to_lower_copy(trim_copy(input));
  if (lowered == "true" || lowered == "yes" || lowered == "on")
  {
    output = true;
    return true;
  }
  if (lowered == "false" || lowered == "no" || lowered == "off")
  {
    output = false;
    return true;
  }

  char* endptr = nullptr;
  const long numeric = std::strtol(lowered.c_str(), &endptr, 10);
  if (endptr != nullptr && *endptr == '\0')
  {
    if (numeric == 0l)
    {
      output = false;
      return true;
    }
    if (numeric == 1l)
    {
      output = true;
      return true;
    }
  }

  return false;
}

bool try_parse_int(const std::string& input, int& output)
{
  const std::string trimmed = trim_copy(input);
  if (trimmed.empty())
    return false;

  char* endptr = nullptr;
  long value = std::strtol(trimmed.c_str(), &endptr, 10);
  if (endptr == nullptr || *endptr != '\0')
    return false;

  if (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min())
    return false;

  output = static_cast<int>(value);
  return true;
}

bool try_parse_double(const std::string& input, double& output)
{
  const std::string trimmed = trim_copy(input);
  if (trimmed.empty())
    return false;

  char* endptr = nullptr;
  double value = std::strtod(trimmed.c_str(), &endptr);
  if (endptr == nullptr || *endptr != '\0')
    return false;

  output = value;
  return true;
}

const std::unordered_map<std::string, std::size_t>& specification_index()
{
  static const std::unordered_map<std::string, std::size_t> index = []()
  {
    std::unordered_map<std::string, std::size_t> map;
    const auto& specs = ConfigurableParameters::specification();
    for (std::size_t i = 0; i < specs.size(); ++i)
    {
      const auto& spec = specs[i];
      map.emplace(spec.qualified_name, i);
      if (spec.alias_unique)
        map.emplace(spec.name, i);
    }

    return map;
  }();

  return index;
}

std::string escape_json_string(const std::string& value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (const char c : value)
  {
    switch (c)
    {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20)
      {
        const char hex_digits[] = "0123456789abcdef";
        escaped += "\\u00";
        escaped += hex_digits[(static_cast<unsigned char>(c) >> 4) & 0x0F];
        escaped += hex_digits[static_cast<unsigned char>(c) & 0x0F];
      }
      else
      {
        escaped += c;
      }
    }
  }
  return escaped;
}

std::string parameter_value_to_string(const ParameterValue& value)
{
  struct ParameterValuePrinter : boost::static_visitor<std::string>
  {
    std::string operator()(int v) const { return std::to_string(v); }
    std::string operator()(double v) const
    {
      std::ostringstream oss;
      oss << std::setprecision(10) << v;
      return oss.str();
    }
    std::string operator()(bool v) const { return v ? "true" : "false"; }
    std::string operator()(const std::string& v) const
    {
      return std::string("\"") + escape_json_string(v) + "\"";
    }
  };

  return boost::apply_visitor(ParameterValuePrinter(), value);
}

void add_parameter_value(dolfin::Parameters& params, const std::string& key, const ParameterValue& value)
{
  struct Visitor : boost::static_visitor<>
  {
    Visitor(dolfin::Parameters& params, const std::string& key) : params(params), key(key) {}

    void operator()(int value) const { params.add(key, value); }
    void operator()(double value) const { params.add(key, value); }
    void operator()(bool value) const { params.add(key, value); }
    void operator()(const std::string& value) const { params.add(key, value); }

    dolfin::Parameters& params;
    const std::string& key;
  } visitor(params, key);

  boost::apply_visitor(visitor, value);
}
}

namespace fem_varden_flow
{
ConfigurableParameters::ConfigurableParameters(const std::string& config_path)
{
  load_defaults();
  if (!config_path.empty())
    load_from_file(config_path);
}

dolfin::Parameters ConfigurableParameters::to_dolfin_parameters(const std::string& name) const
{
  dolfin::Parameters parameters(name);

  for (const auto& entry : values_)
  {
    if (entry.first.find('.') == std::string::npos)
      add_parameter_value(parameters, entry.first, entry.second);
  }

  for (const auto& category : categories())
  {
    if (category.name.empty())
      continue;

    dolfin::Parameters group(category.name);
    for (const auto& spec : category.parameters)
    {
      const std::string qualified = category.name + "." + spec.name;
      const auto it = values_.find(qualified);
      const ParameterValue& value = (it != values_.end()) ? it->second : spec.default_value;
      add_parameter_value(group, spec.name, value);
    }

    parameters.add(group);
  }

  return parameters;
}

dolfin::Parameters ConfigurableParameters::default_category_parameters(const std::string& category_name)
{
  for (const auto& category : categories())
  {
    if (category.name == category_name)
    {
      dolfin::Parameters defaults(category.name);
      for (const auto& spec : category.parameters)
        add_parameter_value(defaults, spec.name, spec.default_value);
      return defaults;
    }
  }

  error("Unknown parameter category '%s'", category_name.c_str());
  return dolfin::Parameters(category_name);
}

const std::vector<ConfigurableParameters::ParameterCategory>& ConfigurableParameters::categories()
{
  static const std::vector<ParameterCategory> groups = {
      {"model_setup",
       "High-level scenario selection and base meshes",
       {
           {"demo_type", ParameterType::String, std::string("ocean"), "Demo selector"},
           {"mesh", ParameterType::String, std::string("default_mesh.xdmf"), "Bulk mesh file"},
           {"surface_mesh", ParameterType::String, std::string("default_surface_mesh.xdmf"), "Surface mesh file"},
           {"kappa_T", ParameterType::Double, 1.41304348e-7, "Thermal diffusivity"},
           {"nu", ParameterType::Double, 1.95e-6, "Kinematic viscosity"},
           {"schmidt_number", ParameterType::Double, 1.0e20, "Schmidt number for variable density (not ocean have to fix this better)"},
           {"rayleigh_number", ParameterType::Double, 0.0, "Rayleigh number for buoyancy-driven cavity benchmark"},
           {"constant_density", ParameterType::Int, 1, "Assume constant density (Boussinesq or constant density incompressible flow). 1 = on, 0 = off (variable density)"},
           {"bad_cells_boundary_condition", ParameterType::Int, 0, "No-slip boundary condition on bad cells. 0 = off, 1 = on, 2 = on with directional smoothing"},
           {"remove_pressure_average", ParameterType::Int, 0, "Remove pressure average. Set to 1 for closed domains. Set to 0 for open domains."},
           {"threshold_bad_cell", ParameterType::Double, 0.9, "Threshold for bad cells. Should not be needed."},

       }},
      {"spatial_discretization",
       "Polynomial degrees and mesh refinement options",
       {
           {"PolDegree", ParameterType::Int, 3, "Polynomial degree of the primary solution space"},
           {"PolDegree_V_M", ParameterType::Int, 3, "Velocity polynomial degree"},
           {"PolDegree_Q", ParameterType::Int, 2, "Pressure and tracer polynomial degree"},
           {"n_points", ParameterType::Int, 0, "Number of sample points"},
           {"number_of_refinements", ParameterType::Int, 0, "Number of uniform mesh refinements"}
       }},
      {"time_integration",
       "Time-step control and integration order",
       {
           {"T", ParameterType::Double, 1.0, "Final simulation time"},
           {"CFL", ParameterType::Double, 0.2, "Courant number controlling the explicit timestep"},
           {"CFL_DG0", ParameterType::Int, 1, "use DG0 representation of h for timestep calculation"},
           {"time_order", ParameterType::Int, 4, "Temporal discretization order. 1 = BDF1, 2 = Crank-Nicolson, 3 = BDF3, 4 = BDF4"},
           {"s_max", ParameterType::Double, 1.01, "Time step increase factor"},
           {"s_min", ParameterType::Double, 0.98, "Time step decrease factor (restarts if step too small)"},
           {"dt_pid_kp", ParameterType::Double, 0.23333333333333334, "PID controller proportional exponent (acts on current normalized error)"},
           {"dt_pid_ki", ParameterType::Double, 0.13333333333333333, "PID controller integral exponent (acts on previous normalized error)"},
           {"dt_pid_kd", ParameterType::Double, 0.03333333333333333, "PID controller derivative exponent (acts on older normalized error)"},
           {"dt_pid_safety", ParameterType::Double, 0.9, "Safety factor multiplying the PID-recommended timestep ratio"},
           {"dt_pid_error_min", ParameterType::Double, 1.0e-6, "Lower clip for normalized timestep error"},
           {"dt_pid_error_max", ParameterType::Double, 1.0e6, "Upper clip for normalized timestep error"},
           {"dt_pid_ratio_min", ParameterType::Double, 0.5, "Minimum ratio allowed between accepted successive timesteps"},
           {"dt_pid_ratio_max", ParameterType::Double, 1.5, "Maximum ratio allowed between successive timesteps"},
           {"dt_pid_retry_ratio_min", ParameterType::Double, 0.05, "Minimum ratio allowed when repeating a rejected timestep"},
           {"bdf_init_factor", ParameterType::Double, 0.1, "Initialization factor for BDF. Important to ensure high-order accuracy at start of simulation or if restarting from a checkpoint. Use larger values to initialize ocean to avoid too small timesteps."}
       }},
      {"stabilization",
       "Residual viscosity parameters",
       {
           {"C_max", ParameterType::Double, 1.0, "Maximum artificial viscosity coefficient"},
           {"C_R", ParameterType::Double, 30.0, "Entropy viscosity scaling coefficient"},
           {"res_scaling_C_l", ParameterType::Double, 1.0, "Residual scaling coefficient"},
           {"C_flat", ParameterType::Double, 1.0, "Flatness threshold for residual computation"},
           {"residual_BDF", ParameterType::Int, 3, "Backward differentiation formula order for residual"},
           {"rv_smoothing_weight", ParameterType::Double, 3.5, "Residual smoothing weight"},
           {"mesh_size_boundary_weight", ParameterType::Double, 0.0, "Weighting of mesh size in boundary cells."},
           {"rv_gamma", ParameterType::Double, 2.0, "Residual gamma parameter"},
           {"directional_wave_speed", ParameterType::Int, 1, "Directional wave speed 1 = on, 0 = off"},
           {"gamma_grad_div", ParameterType::Double, 1.0, "Grad-div penalty parameter (scaled with wavespeed)"},
           {"C_brezzi", ParameterType::Double, 0.0, "Brezzi stabilization coefficient for equal order elements"},
           {"normalization_eps", ParameterType::Double, 1.0e-9, "Normalization epsilon for rv"},
           {"C_max_vms", ParameterType::Double, 0.025, "VMS stabilization coefficient"},
           {"skew_factor", ParameterType::Double, 1.0, "Momentum skew-symmetry factor"},
           {"skew_rho_factor", ParameterType::Double, 0.5, "Tracer skew factor"},
           {"project_uu", ParameterType::Int, 0, "Toggle projection of velocity (use 1 for variable density only)"},
           {"emac_pressure_shift", ParameterType::Double, 0.25, "EMAC pressure shift"},
           {"remove_rho_average", ParameterType::Int, 1, "Remove density average"}
       }},
      {"iterative_solver",
       "Linear and nonlinear solver configuration",
       {
           {"solver_tolerance", ParameterType::Double, 1.0e-8, "Primary solver tolerance"},
           {"solver_tolerance_tracer", ParameterType::Double, 1.0e-10, "Tracer solver tolerance"},
           {"solver_tolerance_LPS", ParameterType::Double, 1.0e-7, "LPS solver tolerance"},
           {"max_solver_iterations", ParameterType::Int, 5000, "Maximum nonlinear iterations"},
           {"monitor_ksp", ParameterType::Int, 0, "Toggle KSP monitor"},
           {"field0_ksp", ParameterType::String, std::string("cg"), "KSP type for velocity block"},
           {"field0_pc", ParameterType::String, std::string("bjacobi"), "Preconditioner for velocity block"},
           {"field1_ksp", ParameterType::String, std::string("cg"), "KSP type for pressure block"},
           {"field1_pc", ParameterType::String, std::string("hypre"), "Preconditioner for pressure block"},
           {"field0_max_iter", ParameterType::Int, 2, "Maximum iterations for field0 solver"},
           {"field1_max_iter", ParameterType::Int, 1, "Maximum iterations for field1 solver"},
           {"field0_tolerance", ParameterType::Double, 1e-8, "Solver tolerance for field0"},
           {"field1_tolerance", ParameterType::Double, 1e-8, "Solver tolerance for field1"},
           {"upper_pc", ParameterType::String, std::string("bjacobi"), "Upper preconditioner type"},
           {"inner_pc", ParameterType::String, std::string("bjacobi"), "Inner preconditioner type"},
           {"schur_type", ParameterType::String, std::string("UPPER"), "Schur complement implementation"},
           {"initial_guess_extrapolation", ParameterType::Int, 1, "Use extrapolation for initial guess when solving linear systems"},
           {"user_schur", ParameterType::Int, 1, "Use user provided Schur complement"},
           {"number_of_newton_iterations", ParameterType::Int, 1, "Maximum Newton iterations"},
           {"C_mass", ParameterType::Double, 1.0e-8, "Schur complement mass matrix coefficient"},
           {"C_laplace", ParameterType::Double, 1.0, "Schur complement stiffness matrix coefficient"}
       }},
      {"boundary_conditions",
       "Coefficients for boundary penalties and restoring",
       {
           {"slip_penalty", ParameterType::Double, 0.0, "Slip boundary condition penalty parameter"},
           {"slip_penalty_fn", ParameterType::Double, 0.0, "Slip boundary condition with facet normal penalty parameter"},
           {"C_ddn_penalty1", ParameterType::Double, 1.0, "Toggle strength of directional do-nothing penalty at backflow. 1.0 = classical DDN, 0.0 = no DDN"},
           {"C_ddn_penalty2", ParameterType::Double, 0.0, "Toggle strength of directional do-nothing penalty at outflow. 0.0 = classical DDN"},
           {"C_slip_friction", ParameterType::Double, -1.5e3, "Slip friction coefficient (used for quadratic drag might remove later)"},
           {"C_nitsche_penalty", ParameterType::Double, 100.0, "Nitsche penalty for weak imposition of Dirichlet BC (currently used in outflow in ocean model)"},
           {"restoring_region_width", ParameterType::Double, 2000.0, "Restoring region width in ocean model. 0.0 = no restoring and Dirichlet BC at outflow"}

       }},
      {"ocean_general",
       "Parameters for ocean setup",
       {
           {"gravitational_acceleration", ParameterType::Double, 9.81, "Gravitational acceleration"},
           {"water_density", ParameterType::Double, 1027.6, "Density of seawater used to compute water pressure in melting"},
           {"T_0", ParameterType::Double, 0.0, "Reference temperature for equation of state"},
           {"S_0", ParameterType::Double, 35.0, "Reference salinity for equation of state"},
           {"ocean_reference_density", ParameterType::Double, 999.8, "Reference density for ocean model. Used to compute rho_boussinesq in code"},
           {"alpha_T", ParameterType::Double, 0.4e-4, "Thermal expansion coefficient"},
           {"beta_S", ParameterType::Double, 0.0008, "Salinity contraction coefficient"},
           {"kappa_S", ParameterType::Double, 8.0180921e-10, "Salinity diffusivity"},
       }},
      {"ocean_melt_model",
       "Parameters for three-equation melt formulations",
       {
           {"melt_with_virtual_flux", ParameterType::Int, 1, "Toggle virtual fluxes for melting (0 = impose normal velocity strongly equal to melt rate and zero tangential velocity, 1 = use virtual fluxes according to Jenkins melt model, 2 = use volume fluxes according to Asay-Davis 2016 eq 31-33)"},
           {"flux_h_factor", ParameterType::Double, 0.1, "Scaling to compute flux when computing melt rate (= 1.0 uses h from solver)"},
           {"melting_with_rv", ParameterType::Int, 1, "Toggle residual viscosity for melting (0 = off, 1 = on with normal velocity, 2 = on with velocity magnitude)"},
           {"smooth_melting", ParameterType::Int, 1, "Toggle melting smoothing. FIXME: remove this one and only use melt_smoothing_weight"},
           {"melt_smoothing_weight", ParameterType::Double, 15.0, "Melting smoothing weight"},
           {"ocean_c_d", ParameterType::Double, 1.5e-3, "Ocean drag coefficient"},
           {"boundary_layer_thickness", ParameterType::Double, 3.333, "Boundary layer thickness in Jenkins 3-equation melt model"},
           {"xi_N", ParameterType::Double, 0.052, "Stability constant used in turbulent exchange closure"},
           {"eta_star", ParameterType::Double, 1.0, "Stability parameter eta*"},
           {"von_karman_constant", ParameterType::Double, 0.40, "Von Kármán constant"},
           {"prandtl_number_exchange", ParameterType::Double, 13.8, "Molecular Prandtl number for exchange velocity"},
           {"schmidt_number_exchange", ParameterType::Double, 2432.0, "Molecular Schmidt number for exchange velocity"},
           {"f_ocean_coriolis", ParameterType::Double, 1.4e-4, "Coriolis parameter in exchange velocity formulation"},
           {"sea_water_kinematic_viscosity", ParameterType::Double, 1.95e-6, "Kinematic viscosity of sea water"},
           {"gammaT_linear", ParameterType::Double, 1.05e-2, "Linear thermal exchange velocity coefficient"},
           {"gammaS_linear", ParameterType::Double, 3.97e-5, "Linear salinity exchange velocity coefficient"},
           {"minimum_farfield_velocity", ParameterType::Double, 1e-3, "Minimum far-field velocity magnitude"},
           {"farfield_drag_coefficient", ParameterType::Double, 1.5e-3, "Momentum exchange coefficient c_d"},
           {"farfield_roughness_length", ParameterType::Double, 3e-4, "Roughness length used if momentum exchange coefficient is zero"},
           {"exchange_velocity_method", ParameterType::String, std::string("full"), "Method for exchange velocity calculation: 'full', 'linear' or 'simple'"},
           {"mixed_layer_heat_capacity", ParameterType::Double, 3974.0, "Specific heat capacity of the mixed layer"},
           {"ice_heat_capacity", ParameterType::Double, 2009.0, "Specific heat capacity of ice"},
           {"latent_heat_of_fusion", ParameterType::Double, 3.34e5, "Latent heat of fusion"},
           {"freezing_equation_salinity_coefficient", ParameterType::Double, -0.0573, "Salinity coefficient in the freezing equation"},
           {"freezing_equation_constant", ParameterType::Double, 0.0939, "Constant coefficient in the freezing equation"},
           {"freezing_equation_pressure_coefficient", ParameterType::Double, -7.53e-8, "Pressure coefficient in the freezing equation"},
           {"mixed_layer_density", ParameterType::Double, 1025.0, "Density of the mixed layer"},
           {"ice_density", ParameterType::Double, 920.0, "Density of ice"},
           {"ice_thermal_conductivity", ParameterType::Double, 1.14e-6, "Molecular thermal conductivity of ice"},
           {"ice_surface_temperature", ParameterType::Double, -20.0, "Temperature at ice shelf surface"},
           {"tidal_velocity_regularization", ParameterType::Double, 0.1, "Tidal velocity regularization parameter"}

       }},
      {"output_diagnostics",
       "Intervals and toggles for writing results",
       {
           {"t_save", ParameterType::Double, 1000.0e10, "Interval for writing solution output"},
           {"t_save_checkpoint", ParameterType::Double, 1000.0e10, "Interval for writing restart checkpoints"},
           {"t_avg_start", ParameterType::Double, 100000.0, "Time to start temporal averaging"},
           {"t_save_avg", ParameterType::Double, 1000.0e10, "Interval for saving averaged fields"},
           {"checkpoint_restart_index", ParameterType::Int, -1, "Checkpoint index to restart from (-1 disables)"},
           {"output_file_type", ParameterType::String, std::string("xdmf"), "Output file format"},
           {"save_reference_solution", ParameterType::Int, 0, "Toggle saving reference solution"},
           {"P3_visualization", ParameterType::Int, 0, "Toggle P3 visualization by refining mesh twice (expensive and takes a lot of memory)"},
           {"smoothing_weight", ParameterType::Double, 0.0, "Smoothing weight to smooth initial condition"}
       }}
  };

  return groups;
}

const std::vector<ConfigurableParameters::ParameterSpec>& ConfigurableParameters::specification()
{
  static const std::vector<ParameterSpec> specs = []() {
    std::vector<ParameterSpec> flattened;
    const auto& groups = categories();

    std::unordered_map<std::string, std::size_t> alias_counts;
    for (const auto& group : groups)
      for (const auto& spec : group.parameters)
        ++alias_counts[spec.name];

    std::size_t total = 0;
    for (const auto& group : groups)
      total += group.parameters.size();
    flattened.reserve(total);

    for (const auto& group : groups)
    {
      const std::string prefix = group.name;
      for (const auto& spec : group.parameters)
      {
        ParameterSpec copy = spec;
        copy.qualified_name = prefix.empty() ? spec.name : prefix + "." + spec.name;
        copy.alias_unique = alias_counts[spec.name] == 1;
        flattened.push_back(copy);
      }
    }
    return flattened;
  }();

  return specs;
}

void ConfigurableParameters::write_default_to(const std::string& file_path)
{
  std::ofstream out(file_path.c_str());
  if (!out.good())
    error("Failed to open default parameter file %s for writing", file_path.c_str());

  out << std::setprecision(6) << std::defaultfloat;

  out << "{\n";

  const auto& groups = categories();
  for (std::size_t i = 0; i < groups.size(); ++i)
  {
    const auto& group = groups[i];
    out << "  \"" << group.name << "\": {\n";

    const auto& specs = group.parameters;
    for (std::size_t j = 0; j < specs.size(); ++j)
    {
      const auto& spec = specs[j];
      out << "    \"" << spec.name << "\": ";

      struct JsonValueWriter : boost::static_visitor<>
      {
        explicit JsonValueWriter(std::ostream& stream) : stream(stream) {}

        void operator()(int value) const { stream << value; }
        void operator()(double value) const
        {
          if (!std::isfinite(value))
            error("Encountered non-finite default parameter value");
          stream << value;
        }
        void operator()(bool value) const { stream << (value ? "true" : "false"); }
        void operator()(const std::string& value) const { stream << '"' << escape_json_string(value) << '"'; }

        std::ostream& stream;
      };

      boost::apply_visitor(JsonValueWriter(out), spec.default_value);

      if (j + 1 < specs.size())
        out << ",";

      out << "\n";
    }

    out << "  }";
    if (i + 1 < groups.size())
      out << ",";
    out << "\n";
  }

  out << "}\n";
}

void ConfigurableParameters::load_defaults()
{
  overridden_parameters_.clear();
  unknown_parameters_.clear();
  current_source_.clear();
  for (const auto& spec : specification())
  {
    set_value(spec.qualified_name, spec.default_value);
    if (spec.alias_unique)
      set_value(spec.name, spec.default_value);
  }
}

void ConfigurableParameters::load_from_file(const std::string& file_path)
{
  overridden_parameters_.clear();
  unknown_parameters_.clear();
  current_source_ = file_path;

  std::string extension;
  const auto dot = file_path.find_last_of('.');
  if (dot != std::string::npos)
  {
    extension = file_path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  }

  if (extension == ".json")
    load_from_json(file_path);
  else
    load_from_legacy_format(file_path);

  report_defaults_used(file_path);
  current_source_.clear();
}

void ConfigurableParameters::load_from_json(const std::string& file_path)
{
  boost::property_tree::ptree tree;
  try
  {
    boost::property_tree::read_json(file_path, tree);
  }
  catch (const std::exception& e)
  {
    error("Failed to read parameters from %s: %s", file_path.c_str(), e.what());
  }

  std::function<void(const boost::property_tree::ptree&, const std::string&)> parse =
      [&](const boost::property_tree::ptree& node, const std::string& prefix)
  {
    for (const auto& child : node)
    {
      if (child.first.empty())
        error("Array-valued parameters are not supported (context: %s)", prefix.c_str());

      const std::string full_key = prefix.empty() ? child.first : prefix + "." + child.first;

      if (child.second.empty())
      {
        apply_leaf(full_key, child.second.data());
      }
      else
      {
        const bool contains_array = child.second.count("") > 0;
        if (contains_array)
          error("Array-valued parameters are not supported (key: %s)", full_key.c_str());

        parse(child.second, full_key);
      }
    }
  };

  parse(tree, std::string());
}

void ConfigurableParameters::load_from_legacy_format(const std::string& file_path)
{
  std::ifstream in(file_path.c_str());
  if (!in.good())
    error("The parameters file %s cannot be opened", file_path.c_str());

  std::string line;
  std::size_t line_number = 0;
  while (std::getline(in, line))
  {
    ++line_number;

    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos)
      line = line.substr(0, comment_pos);

    std::istringstream ss(line);
    std::string value_token;
    std::string key_token;
    if (!(ss >> value_token))
      continue;

    if (!(ss >> key_token))
      error("Malformed parameter line %zu in %s", line_number, file_path.c_str());

    apply_leaf(key_token, value_token);
  }
}

void ConfigurableParameters::apply_leaf(const std::string& key, const std::string& raw_value)
{
  const ParameterSpec* spec = find_spec(key);
  const std::string trimmed_value = trim_copy(raw_value);

  if (!spec)
  {
    if (unknown_parameters_.insert(key).second)
    {
      std::ostringstream message;
      if (!current_source_.empty())
        message << "Parameter '" << key << "' in " << current_source_ << " is not recognized; ignoring value '" << trimmed_value << "'";
      else
        message << "Parameter '" << key << "' is not recognized; ignoring value '" << trimmed_value << "'";

      info(message.str());
    }
    return;
  }

  const std::string& canonical_key = spec->qualified_name;
  overridden_parameters_.insert(canonical_key);
  const ParameterValue parsed_value = parse_with_type(canonical_key, spec->type, trimmed_value);
  set_value(canonical_key, parsed_value);
  if (spec->alias_unique)
    set_value(spec->name, parsed_value);
}

void ConfigurableParameters::set_value(const std::string& key, const ParameterValue& value)
{
  values_[key] = value;
}

ConfigurableParameters::ParameterValue ConfigurableParameters::parse_with_type(const std::string& key, ParameterType type, const std::string& raw_value) const
{
  switch (type)
  {
  case ParameterType::Int:
  {
    int value = 0;
    if (!try_parse_int(raw_value, value))
      error("Failed to parse integer parameter '%s' with value '%s'", key.c_str(), raw_value.c_str());
    return value;
  }
  case ParameterType::Double:
  {
    double value = 0.0;
    if (!try_parse_double(raw_value, value))
      error("Failed to parse floating-point parameter '%s' with value '%s'", key.c_str(), raw_value.c_str());
    return value;
  }
  case ParameterType::Bool:
  {
    bool value = false;
    if (!try_parse_bool(raw_value, value))
      error("Failed to parse boolean parameter '%s' with value '%s'", key.c_str(), raw_value.c_str());
    return value;
  }
  case ParameterType::String:
    return raw_value;
  }

  error("Unknown parameter type");
  return ParameterValue();
}

const ConfigurableParameters::ParameterSpec* ConfigurableParameters::find_spec(const std::string& key) const
{
  const auto& index = specification_index();
  const auto it = index.find(key);
  if (it == index.end())
    return nullptr;

  const auto& specs = specification();
  return &specs[it->second];
}

void ConfigurableParameters::report_defaults_used(const std::string& source) const
{
  const auto& specs = specification();
  for (const auto& spec : specs)
  {
    if (overridden_parameters_.count(spec.qualified_name) != 0)
      continue;

    const auto value_it = values_.find(spec.qualified_name);
    if (value_it == values_.end())
      continue;

    if (value_it->second != spec.default_value)
      continue;

    const std::string default_value = parameter_value_to_string(value_it->second);
    std::ostringstream message;
    if (!source.empty())
      message << "Parameter '" << spec.qualified_name << "' not provided in " << source << "; using default value " << default_value;
    else
      message << "Parameter '" << spec.qualified_name << "' not provided; using default value " << default_value;

    info(message.str());
  }
}
}
