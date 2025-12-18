// Copyright (C) 2023 Christian Helanow
// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
//

#ifndef __EXCHANGEVELOCITY_H
#define __EXCHANGEVELOCITY_H

// dolfin
#include <dolfin/parameter/Parameters.h>

// general
#include <algorithm>

#include <ConfigurableParameters.h>

namespace fem_varden_flow
{
  /// This class computes the thermal (gammaT) and salinity (gammaS)
  /// exchange velocity as functions of a given far-field velocity
  /// (speed), u_star, as presented in the Holland and Jenkins (1999):
  /// Modeling Thermodynamic Ice–Ocean Interactions at the Base of an
  /// Ice Shelf (HJ99).

  /// In this paper three methods to compute the exchange velocities
  /// are given, which in this class are denoted as "full", "simple"
  /// and "linear".

  /// "full" method:
  /// In this implementation the exchange velocities depend on both
  /// the molecular and turbulent transfer parameters. The relation(s)
  /// for gammaT and gammaS are presented in Eq. 14 in the HJ99,
  /// with additional variables/relations defined/specified in Eqs
  /// 15-18. This is the "default" implementation (using the global
  /// configuration unless otherwise explicitly specified by keyword
  /// arguments) which generally follows MIT-GCM. See:
  /// http://lxr.mitgcm.org/lxr2/source/MITgcm/pkg/shelfice/shelfice_thermodynamics.F#0471

  /// "simple" method:
  /// In this implementation the exchange velocities depend on the
  /// molecular transfer parameter and a *simplified* expression of
  /// the turbulent transfer parameter in which the effects of the
  /// Coriolis parameter (and more) are ignored. In detail, the first
  /// term in Eq. 15 (together with Eqns. 17 and 18) are skipped,
  /// making the expression for the turbulent transfer parameter
  /// independent of the far-field velocity. In HJ99
  /// the molecular processes are demonstraded to be of much greater
  /// importance than the turbulent processes.

  /// "linear" method:
  /// The *parametrized* versions are implemented with this method.
  /// The parameterization can be found in HJ99 section 3: Comparison
  /// of model results. Essentially, for a typical range of far-field
  /// velocities, u_star \in (0, 0.02) m/s, the value of
  /// gammaT increases (rougly) linearly with u_star.
  /// The ratio gammaT/gammaS is also (rougly)
  /// constant, indicating that also gammaS is a linear
  /// function of u_star.
  class ExchangeVelocity
  {
  public:
    /// Default constructor
    /// initializes using default parameters as specified by
    /// fem_varden_flow::ConfigurableParameters.
    ExchangeVelocity();

    /// Constructor
    /// initializes using specified parameters (updating the default
    /// parameters)
    ExchangeVelocity(const dolfin::Parameters& p);
    /// Destructor
    ~ExchangeVelocity();

    /// Computes the thermal exchange velocity, gammaT
    double gammaT(double u_star);

    /// Computes the salinity exchange velocity, gammaS
    double gammaS(double u_star);

    /// Updates the parameter set
    void update_parameters(const dolfin::Parameters& parameters);

    /// Return the default parameters
    static dolfin::Parameters default_parameters()
    {
      return fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_melt_model");
    }

    /// Parameters describing the exchange velocity closure
	dolfin::Parameters parameters{default_parameters()};

    /// Avilable methods for this class
    static const std::vector<std::string> valid_methods;

  private:
    // the method used
    std::string m_method;

    // check if given method is avlailable/valid
    bool is_valid_method(std::string method)
    {
      return std::find(valid_methods.begin(),
                       valid_methods.end(),
                       method) != valid_methods.end();
    }

	/// Return the parameters relevant for ExchangeVelocity from a given parameter set

	// 	CH: not sure if this is a good or a bad idea: removing the
	// parameters that are unused in this class. Pro: a user would
	// be notified if trying to update the parameter set with a
	// parameter that is not used in this class. Con:
	// maintainability.

	// static dolfin::Parameters create_local_parameters(dolfin::Parameters& p)
	//   {
	// 	dolfin::Parameters p_loc("exchangevelocity_parameters");
	// 	// p_loc.rename("exchangevelocity_parameters");
	// 	// p_loc.remove("UM_min");
	// 	// p_loc.remove("boundary_layer_thickness");
	// 	return p_loc;
	//   }

    // Convenience variable functions and variables: necessary? Could
    // compute in each function call rather than precompute and
    // update.
    
    // computes and sets the (private) convenience parameters used for
    // computing gammaT and gammaS from the (current) parameter set:
    // GammaMoleT, GammaMoleS, GammaTurbConst, GammaCoriolisFactor,
    // GammaCoriolisConst, GammaT_linear, GammaS_linear
    void compute_gamma_variables();
    

    // Molecular exchange temperature: Holland/Jenkins Equation 16
    double GammaMoleT;

    // Molecular exchange salinity: Holland/Jenkins Equation 16
    double GammaMoleS;

    // Turbulent exchange constant
    double GammaTurbConst;

    // Coriolis expression prefactor
    double GammaCoriolisFactor;

    // Coriolis expression constant
    double GammaCoriolisConst;

    // Linear relation prefactor. If method is "linear":
    // gammaT = GammaT_linear * u_star
    double GammaT_linear;

    // Linear relation prefactor. If method is "linear":
    // gammaS = GammaS_linear * u_star
    double GammaS_linear;
  };
}
#endif
