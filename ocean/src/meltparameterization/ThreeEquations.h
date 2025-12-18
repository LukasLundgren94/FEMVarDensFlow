// Copyright (C) 2024 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __THREE_EQUATIONS_H
#define __THREE_EQUATIONS_H

// dolfin
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>

#include <dolfin/parameter/Parameters.h>


#include <ConfigurableParameters.h>
#include "ExchangeVelocity.h"

namespace fem_varden_flow
{
  /// Solves the three-equation system for the boundary melt rate,
  /// salinity and temperature
  ///
  /// The three-equation system is described in:
  /// - Kimura et al. (2013): Adaptation of an unstructured-mesh,
  ///   ﬁnite-element ocean model to the simulation of ocean circulation
  ///   beneath ice shelves.
  /// - Holland and Jenkins (1999): Modeling Thermodynamic Ice–Ocean
  ///   Interactions at the Base of an Ice Shelf.
  ///
  /// The three relevant equations (e.g, eqns (5-7) in Kimura et
  /// al., 2013) are coupled and lead to a quadratic equation for
  /// the boundary salinity. Only one of the two possible values for
  /// the boundary salinity is positive definite (and considered in
  /// here to be the only value of physical importance). The
  /// computed value for the boundary salinity is then used to solve
  /// for the boundary temperature and finally the melt rate.
  ///
  /// These (point-wise) variables/values can be used e.g., when
  /// computing the salinity flux and temperature flux, as described
  /// by eqns (8) and (9) in Kimura et al., 2013 (after Jenkins et
  /// al., 2001).
  class ThreeEquations
  {
  public:
	/// Default constructor
	/// uses three-equation parameters specified in
	/// fem_varden_flow::ConfigurableParameters (category "ocean_melt_model")
	/// *Arguments*
	/// exchange_velocity (_ExchangeVelocity_)
	///   Instance specifying how to compute thermal and salinity
	///   exchange velocities (gammaS and gamma T in
	///   Holland/Jenkins (1999)). See fem_varden_flow::ExchangeVelocity.

	ThreeEquations(const dolfin::Parameters& p);

	/// Destructor
	~ThreeEquations();

	/// Solves the three-equation system
	/// *Arguments*
	/// Sm (_double_)
	///   The far-field salinity value.
	/// Tm (_double_)
	///   The far-field temperature value.
	/// P (_double_)
	///   The "far-field" pressure value. Note: this is computed
	///   differently by different people, e.g., the pressure
	///   field as solved by the Boussinesqu eq at the boundary,
	///   the pressure field as a far-field value or just
	///   hydrostatic pressure.
	/// gammaS
	///   Value of salinity exchange velocity (gammaS in
	///   Holland/Jenkins (1999)) as computed as a function of the
	///   far-field velocity. See fem_varden_flow::ExchangeVelocity.gammaS().
	/// gammaT
	///   Value of thermal exchange velocity (gammaT in
	///   Holland/Jenkins (1999)) as computed as a function of the
	///   far-field velocity. See fem_varden_flow::ExchangeVelocity.gammaT().
	/// return_negative_root (_bool_)
	///   If true, returns the smallest root. For the purpose of
	///   the three-equation system, the poisitive definite
	///   solution is always sought, e.g., this parameter shoud be
	///   false.
	///
	/// *Returns*
	///   vector containing (boundary salinity, boundary temperature, boundary melt rate)
	std::vector<double> solve(double Sm, double Tm, double P, double gammaS, double gammaT,
							  bool return_negative_root=false);

	/// Return the default parameters (as defined by
	/// fem_varden_flow::ConfigurableParameters)
	static dolfin::Parameters default_parameters()
	  {
		return fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_melt_model");
	  }

	/// Parameters describing the three-equation melt closure
	dolfin::Parameters parameters;
		
	/// Updates the parameter set
	void update_parameters(const dolfin::Parameters& parameters);

  private:
	// Three-equation variable holders
	double c_I_;
	double c_M_;
	double a_;
	double b_;
	double c_;
	double L_;
	double Ts_;
  };
}

#endif
