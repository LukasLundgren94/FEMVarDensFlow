// Copyright (C) 2024 Christian Helanow
// Copyright (C) 2025 Lukas Lundgren
// Copyright (c) 2025 Jonathan Wiskandt
//
// Licensed under the MIT License – see the LICENSE file for details
//
// First added:  2024-01-07

// CH: Right now the boundary normal argument is a
// fem_varden_flow::NodeNormal, which is evaluated on the *whole*
// boundary. This seems unnecessary, but right now it is easier to do
// this implementation wise. Really, the whole NodeNormal class
// should be restructured, so that it is possible to define it for a
// subset (facet markers) of the boundary in different priority
// (contining normal on only parts of the boundary, with relevant dofs
// --> instead of Expressions this class could be a Function that can
// be updated for only the select relevant dofs.)

// dolfin
#include <dolfin/mesh/Mesh.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
#include <dolfin/log/log.h>

// #include <dolfin/function/GenericFunction.h>
// #include <dolfin/function/Expression.h>
// #include <dolfin/common/Array.h>
// #include <dolfin/mesh/Cell.h>
// #include <dolfin/mesh/Mesh.h>
#include <dolfin/function/FunctionSpace.h>

// general
#include <cmath>

// fem_varden_flow
#include "FrictionVelocity.h"
#include "NodeNormal.h"

using namespace fem_varden_flow;

FrictionVelocity::FrictionVelocity(std::shared_ptr<dolfin::Function> velocity,
								   fem_varden_flow::NodeNormal& normal,
								   const dolfin::Parameters& p,
								   double zero_normal_tolerance) :
  dolfin::Expression()
{
  if (velocity->value_rank() != 1)
  {
	dolfin::error("velocity function must be a vector function.");
  }
  else if (velocity->value_dimension(0) != velocity->function_space()->mesh()->geometry().dim())
	
  {
	dolfin::error("vector dimension must be same as mesh dimension.");
  }
  velocity_variable = velocity;
  velocity_variable->set_allow_extrapolation(true);
  update_parameters(p);
  boundary_normal = normal.normal();
  boundary_normal.set_allow_extrapolation(true);
  n_tol_ = zero_normal_tolerance;
  dim_ = velocity->value_dimension(0);
}


FrictionVelocity::~FrictionVelocity()
{
  // Do nothing
}

void FrictionVelocity::update_parameters(const dolfin::Parameters& parameters)
{
  this->parameters = parameters;
  bl_thick_ = this->parameters["boundary_layer_thickness"];
  c_d_ = this->parameters["farfield_drag_coefficient"];
  try
  {
    k = this->parameters["von_karman_constant"];
  }
  catch (...)
  {
    auto exchange_defaults =
        fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_exchange_velocity");
    k = exchange_defaults["von_karman_constant"];
  }
  z_0 = this->parameters["farfield_roughness_length"];
  if (c_d_ == 0.0)
  {
    set_log_active(true);
    dolfin::info("c_D set to zero, use Law of the Wall with roughness length z_0 = %e and BL-Thickness = %f m",
                 z_0, bl_thick_);
    set_log_active(false);
    c_d_ = std::pow(k / std::log(bl_thick_ / z_0), 2);
    set_log_active(true);
    dolfin::info("c_D  = %e", c_d_);
    set_log_active(false);
  }
  UM_min_ = this->parameters["minimum_farfield_velocity"];
}

void test_val(const dolfin::Array<double>& a, std::string s)
{
  std::cout << s << " values: ( ";
  for (std::size_t i = 0; i<a.size(); ++i)
  {
	std::cout << a[i] << " ";
  }
  std::cout << ")\n";
	
}
	
void test_val(double a, std::string s)
{
  std::cout << s << " value: " << a;
}
	

void FrictionVelocity::eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const
{
  // if magnitude of normal is not (near) 1.0, the point should not
  // be on any part of the boundary for which the far-field
  // variable is to be evaluated, so return zero
  // Note: this is based on the assumption that the normal is a
  // CG1 function and normalized everywhyere on the boundary: any
  // value less/other than |n| = 1 should indicate that the
  // normal is evaluated away from the boundary.
	
  // CH: should I assert sizes here?
	
  dolfin::Array<double> normal_pointval(boundary_normal.value_size());
  try{
  boundary_normal.eval(normal_pointval, x);
  }
  catch (std::exception& e)
  {
    dolfin::warning("FrictionVelocity: Error in boundary_normal.eval().");
    array_fill(values, NAN);
    return;
  }

  // test_val(normal_pointval, "normal_pointval");

  if (dolfin::near(array_norm(normal_pointval), 0.0, n_tol_))
  {

	dolfin::warning("FrictionVelocity: Normal magnitude < tolerance: setting value to zero.");
	array_fill(values, NAN);
	return;
  }
  // get value at far-field position: outward-pointing boundary
  // normal, so minus is for into the domain
  dolfin::Array<double> farfield_pos(dim_);

  for (std::size_t i = 0; i < farfield_pos.size(); ++i)
  {
	farfield_pos[i] = x[i] - bl_thick_ * normal_pointval[i];
  }
  // test_val(x, "x");
  // test_val(farfield_pos, "farfield_pos");

  // eval the farifield variable at the position
  dolfin::Array<double> velocity_pointval(dim_);
  // Lukas: try to comment out this line
  try{
  velocity_variable->eval(velocity_pointval, farfield_pos);
  }
  catch (std::exception& e)
  {
    dolfin::warning("FrictionVelocity: Error in velocity_variable.eval().");
    array_fill(values, NAN);
    return;
  }
  // magnitude of velocity at farifield position
  double vel_mag = array_norm(velocity_pointval);
  // magnitude of velocity in normal direction at farfield position
  double vel_n_mag{0.0};
  for (std::size_t i = 0; i < dim_; ++i)
  {
	vel_n_mag += velocity_pointval[i]*normal_pointval[i];
  }
  vel_n_mag = std::abs(vel_n_mag);
  // magnitude of tangential part of velocity at farfield position
  double vel_t_mag_sq = vel_mag*vel_mag - vel_n_mag*vel_n_mag;
  // warnings or errors if negative values
  if (dolfin::near(vel_t_mag_sq, 0))
  {
	vel_t_mag_sq = 0.0;
  }
  else if (vel_t_mag_sq < 0.0)
  {
	std::string msg("Negative squared tangential velocity encountered at coordinate x = ( ");
	for (std::size_t i=0 ; i < x.size() ; ++i)
	{
	  msg += std::to_string(x[i]) + " ";
	}
	msg += " )\n";
	// dolfin::error(msg);
  // Lukas:
  // set_log_active(true);
  dolfin::info(msg);
  // std::cout << "msg" <<msg << std::endl; 
  // set_log_active(false);
  vel_t_mag_sq = 0.0;
  array_fill(values, NAN);
    return;
  }	
  // get friction velocity, u*: i.e., conditional tangential boundary
  // velocity (speed) time momentum exchange coeff.
	// if c_d_ is zero, set to function of bl_thick_
  values[0] = std::sqrt(c_d_) * std::max(std::sqrt(vel_t_mag_sq), UM_min_);
}
