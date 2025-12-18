// Copyright (C) 2024 Christian Helanow
// Copyright (c) 2025 Jonathan Wiskandt
// Licensed under the MIT License – see the LICENSE file for details
//

#ifndef __FRICTIONVELOCITY_H
#define __FRICTIONVELOCITY_H

// dolfin
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>

#include <dolfin/parameter/Parameters.h>


#include <ConfigurableParameters.h>
#include "NodeNormal.h"


namespace fem_varden_flow
{
	/// Expression for the friction velocity at the ice-ocean interface.
	///
	/// The friction velocity, u_*, is defined as (from Jenkins and Holland, 1999)
	///
	///    u_*^2 = c_d * U_M^2
	///
	/// where :math:`c_d` is a dimensionless drag coefficient (as
	/// defined in `feffii.parameters.config['3eqs']['c_d']`), with
	/// default value :math:`c_d = 1.5 \times 10^{-3}`, and the
	/// velocity of the mixed layer :math:`U_M` the ice being
	/// considered stationary.
	/// From the above, this expression is a "speed", rather than a velocity.
	///
	/// The variable U_M is the far-field velocity and is in
	/// this Expression calculated after Kimura (2013), as speed of
	/// ocean ﬂow oriented parallel to the ice shelf at a
	/// boundary-layer distance away (in the normal direction) from
	/// the ice shelf. The distance is specified in
	/// fem_varden_flow::ConfigurableParameters (category "ocean_melt_model")
	/// parameter "boundary_layer_thickness"
	///
	/// Following the MIT-GCM model, a *minimum value* for
	/// :math:`u_*^2 = c_d 10^{-6}`, i.e. if the far-field velocity
	/// :math:`U_M` is too low, a small value will be used regardless.
	/// This is so that some melting in the three-equation system
	/// occurs even if ther for instance is a zero velocity field.
	///
	/// See:
	///
    /// - Holland and Jenkins (199): Modeling Thermodynamic Ice–Ocean
    ///   Interactions at the Base of an Ice Shelf.
    /// - Kimura et al. (2013): Adaptation of an unstructured-mesh,
    ///   ﬁnite-element ocean model to the simulation of ocean circulation
    ///   beneath ice shelves.
    /// - http://lxr.mitgcm.org/lxr2/source/MITgcm/pkg/shelfice/shelfice_thermodynamics.F
    ///   Lines: 466-467

	
	class FrictionVelocity : public dolfin::Expression
	{
	public:
		/// Constructor
		/// V_CG1 necessary if higher-order function space is used for
		/// `velocity`, due to initiation of NodeNormal
		FrictionVelocity(std::shared_ptr<dolfin::Function> velocity,
					   fem_varden_flow::NodeNormal& normal,
					   const dolfin::Parameters& p,
					   double zero_normal_tolerance=1e-12);

		/// Destructor
		~FrictionVelocity();

		/// Evaluates the friction velocity at point x.
		///
		/// Since the normal at the point is readily available, the
		/// magnitude of the tangential part of the far-field velocity
		/// is computed using the Pythagorean theorem as:
		///
        ///    \|u_t\| = \sqrt{\|u\|^2 - \|u_n\|^2}
        ///    u_n = (u\cdot n) n
        ///    \|n\| = 1
		virtual void eval(dolfin::Array<double>& values,
						  const dolfin::Array<double>& x) const override;
		
		/// Return the default parameters (as defined by
		/// fem_varden_flow::ConfigurableParameters)
		static dolfin::Parameters default_parameters()
		{
			dolfin::Parameters defaults =
			    fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_melt_model");
			auto exchange_defaults =
			    fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_melt_model");
			const double von_karman = static_cast<double>(exchange_defaults["von_karman_constant"]);
			defaults.add("von_karman_constant", von_karman);
			return defaults;
		}

		/// Parameters describing far-field sampling and friction velocity
		dolfin::Parameters parameters;
		
		/// Updates the parameter set
		void update_parameters(const dolfin::Parameters& parameters);
		
		/// Field that will be evaluated at boundary layer thicknes away
		/// from mesh boundary
		std::shared_ptr<dolfin::Function> velocity_variable;

		/// Specified boundary normal
		// should this be a shared pointer or just function
		// std::shared_ptr<dolfin::Function> boundary_normal;
		dolfin::Function boundary_normal;
		
	private:
		/// Return the parameters relevant for FrictionVelocity as
		// defined in the global fem_varden_flow::ocean_parameter
		// static dolfin::Parameters create_local_parameters(dolfin::Parameters& p)
		// 	{
		// 		double blt = p["boundary_layer_thickness"];
		// 		dolfin::Parameters p_loc("farfieldvelocity_parameters");
		// 		p_loc.add("boundary_layer_thickness", blt);
		// 		return p_loc;
		// 	}
		
		// Convenience functions for dolfin::Array class
		double array_norm(const dolfin::Array<double>& a) const
			{
				double vec_norm = 0.0;
				for (std::size_t i = 0; i < a.size(); ++i)
				{
					vec_norm += a[i]*a[i];
				}
				return std::sqrt(vec_norm);
			}

		void array_fill(dolfin::Array<double>& a, double val) const
		{
			for (std::size_t i = 0; i < a.size(); ++i)
				{
					a[i] = val;
				}
			
		}
		
		// Variables holding values specified in parameters or class arguments
		double bl_thick_;
		double c_d_;
		double UM_min_;
		double n_tol_;
		double k; // von Karman constant
		double z_0; // roughness length
		std::size_t dim_;
	};
}

#endif
