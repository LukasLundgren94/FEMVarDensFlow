// Copyright (C) 2024 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details
//
// First added:  2024-01-29

// CH: Right now the boundary normal argument is a
// fem_varden_flow::UnitNodeNormal, which is evaluated on the *whole*
// boundary. This seems unnecessary, but right now it is easier to do
// this implementation wise. Really, the whole UnitNodeNormal class
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
#include <complex>

// fem_varden_flow
#include "ExchangeVelocity.h"
#include "ThreeEquations.h"

using namespace fem_varden_flow;

ThreeEquations::ThreeEquations(const dolfin::Parameters& p)
{
	update_parameters(p);
}

ThreeEquations::~ThreeEquations()
{
	// Do nothing
}

void ThreeEquations::update_parameters(const dolfin::Parameters& parameters)
{
  this->parameters = parameters;
  c_I_ = this->parameters["ice_heat_capacity"];
  c_M_ = this->parameters["mixed_layer_heat_capacity"];
  a_ = this->parameters["freezing_equation_salinity_coefficient"];
  b_ = this->parameters["freezing_equation_constant"];
  c_ = this->parameters["freezing_equation_pressure_coefficient"];
  L_ = this->parameters["latent_heat_of_fusion"];
  Ts_ = this->parameters["ice_surface_temperature"];
}

std::vector<double> ThreeEquations::solve(double Sm, double Tm, double P, double gammaS,
										  double gammaT,
										  bool return_negative_root)
{
	// Specify the constants in the polynomial to solve for Sb, boundary salinity
	double A = (c_I_*a_ - c_M_*gammaT*a_/gammaS);
	double B = (L_ + c_I_*(b_ + c_*P - Ts_)
				+ c_M_*gammaT/gammaS * (Tm - b_ - c_*P)
				- c_I_*a_*Sm);
	double C = -(L_ + c_I_*(b_ + c_*P - Ts_))*Sm;
	double Sb;
	// solve quadratic equation for Sb, boundary salinity
	if (return_negative_root)
	{
		Sb = (-B - std::sqrt(B*B - 4*A*C)) / (2*A);
	}
	else
	{
		Sb = (-B + std::sqrt(B*B - 4*A*C)) / (2*A);
	}
	// Solve for Tb, boundary temperature
	double Tb = a_*Sb + b_ + c_*P;
    // Solve for mb, boundary melt rate
	double mb = (c_M_*gammaT * (Tm - (a_*Sb + b_ + c_*P))
				 / (L_ + c_I_*(a_*Sb + b_ + c_*P - Ts_)));
	
	
	// Return values
	return std::vector<double>{Sb, Tb, mb};
}       