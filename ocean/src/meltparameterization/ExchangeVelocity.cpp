// Copyright (C) 2023 Christian Helanow
// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
//
// First added:  2023-12-22

// CH: May be worth to refactor into separate classes for perfomance.
// However, not sure if this would be the case: it seems like the
// compiler manages to optimize away this simple branching.

// dolfin
#include <dolfin/log/log.h>
#include <dolfin/parameter/Parameters.h>

// general
#include <cmath>
#include <iostream>

// fem_varden_flow
#include "ExchangeVelocity.h"

using namespace fem_varden_flow;


ExchangeVelocity::ExchangeVelocity() : ExchangeVelocity(dolfin::Parameters{})
{
  // Do nothing
}


ExchangeVelocity::ExchangeVelocity(const dolfin::Parameters& p)
{

  // Update parameters and compute the relevant variables
  update_parameters(p);
}

// ExchangeVelocity::ExchangeVelocity(std::string method) : m_method{method}
// {
//   std::cout << "Calling ExchangeVelocity(" << "\"" << m_method << "\")\n"; 
//   if (!is_valid_method(m_method))
//     {
//       dolfin::error("ExchangeVelocity.cpp:: "
//                     "unknown exchangevelocity method \"%s\"."
//                     "Use exchangevelocity::valid_methods() to "
//                     "see available methods.",
//                     m_method.c_str());
      
//     }
// }


// ExchangeVelocity::ExchangeVelocity(const dolfin::Parameters& p,
//                                    std::string method) :
//   ExchangeVelocity(method)
// {
//   parameters.update(p);
// }

ExchangeVelocity::~ExchangeVelocity()
{
  // Do nothing
}

void ExchangeVelocity::update_parameters(const dolfin::Parameters& parameters)
{
  this->parameters = parameters;
  std::string const m_method_temp = parameters["exchange_velocity_method"];
  m_method = m_method_temp;
  if (!is_valid_method(m_method))
    {
      dolfin::error("ExchangeVelocity.cpp:: "
                    "unknown exchangevelocity method \"%s\"."
                    "Use exchangevelocity::valid_methods() to "
                    "see available methods.",
                    m_method.c_str());
      
    }
  compute_gamma_variables();
}

double ExchangeVelocity::gammaT(double u_star)
{
  // Returns the result of Equation 15 in Jenkins/Holland (1999), or a
  // simplification
  // Note: the simplification for method=="simple" depends on the set
  // parameters GammaCoriolisConst and GammaCoriolisFactor, and are
  // set depening on method when updating the parameter set.
  if (m_method == "linear")
    {
      return GammaT_linear * u_star;
    }
  else
    {
      // Could possibly be undefined/throw error if u_star<0, which it
      // should not be. Is it better to "hide" such an error using
      // log(u_star^2) instead of the computed 2*log(u_star)
      return u_star /
             (GammaTurbConst
              + GammaCoriolisFactor * std::log(u_star)
              + GammaCoriolisConst
              + GammaMoleT);
    }
}

double ExchangeVelocity::gammaS(double u_star)
{
  // Returns the result of Equation 15 in Jenkins/Holland (1999), or a
  // simplification
  // Note: the simplification for method=="simple" depends on the set
  // parameters GammaCoriolisConst and GammaCoriolisFactor, and are
  // set depening on method when updating the parameter set.
  if (m_method == "linear")
    {
      return GammaS_linear * u_star;
    }
  else
    {
      // Could possibly be undefined/throw error if u_star<0, which it
      // should not be. Is it better to "hide" such an error using
      // log(u_star^2) instead of the computed 2*log(u_star)
      return u_star /
             (GammaTurbConst
              + GammaCoriolisFactor * std::log(u_star)
              + GammaCoriolisConst
              + GammaMoleS);
    }
}

const std::vector<std::string> ExchangeVelocity::valid_methods{
  "full", "simple", "linear"};

void ExchangeVelocity::compute_gamma_variables()
{
  double Pr = parameters["prandtl_number_exchange"];
  double Sc = parameters["schmidt_number_exchange"];
  double xi_N = parameters["xi_N"];
  double eta_star = parameters["eta_star"];
  double k = parameters["von_karman_constant"];
  double f = parameters["f_ocean_coriolis"];
  double nu_sw = parameters["sea_water_kinematic_viscosity"];
  
  // Equation 16 in Jenkins/Holland (1999)
  GammaMoleT = 12.5 * std::pow(Pr, 2.0/3) - 6.0;
  GammaMoleS = 12.5 * std::pow(Sc, 2.0/3) - 6.0;
  // Equation 15 in Jenkins/Holland (1999): last two terms which gives
  // the constant part of the turbulent gamma expression.
  GammaTurbConst = 1/(2 * xi_N * eta_star) - 1/k;

  // Equation 15 in Jenkins/Holland (1999) giving the part of the
  // turbulent gamma expression depending on the Coriolis parameter:
  // here the equation is rearranged (using Equation (17))so that the
  // two factors below can easily be used to compute the complete
  // expression (dependent on u_star)
  // 1/k * ln((u_star^2 * xi_N * eta_star^2) / (|f * 5 * nu_sw|))
  // becomes
  // 2/k * ln(u_star) + 1/k * ln((xi_N * eta_star^2) / (|f * 5 * nu_sw|))
  // Note: there is not an abs value of 'f' in the paper: however, I
  // don't see how this would work if 'f' has a negative value (which
  // it is specified to have, if not zero (?))
  // Note: this could probably be better done with (maybe) separate
  // classes for each method.
  // Note: if u_star<0 (which it should never be), this would pass in
  // the original formulation (log(u_star^2)) but fail in the
  // refactorization (2*log(u_star))
  
  // This should be safe given that 'f' is a set parameter.
  // If f is zero skip the Coriolis term altogether (after MIT-gcm).
  if (!(m_method == "full") || f == 0.0)
    {
      GammaCoriolisFactor = 0.0;
      GammaCoriolisConst = 0.0; 
    }
  else
    {
      GammaCoriolisFactor = 2.0/k;
      GammaCoriolisConst = 1.0/k *
                           (std::log((xi_N * eta_star * eta_star) /
                                     (std::abs(f * 5 * nu_sw))));
    }

  // Factors for linear relation
  GammaT_linear = parameters["gammaT_linear"];
  GammaS_linear = parameters["gammaS_linear"];
}
