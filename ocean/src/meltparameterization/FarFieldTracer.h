// Copyright (C) 2023 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details
//

#ifndef __FARFIELDTRACER_H
#define __FARFIELDTRACER_H

// dolfin
#include <dolfin/function/Function.h>
#include <dolfin/function/Expression.h>

#include <dolfin/parameter/Parameters.h>


#include <ConfigurableParameters.h>
#include "NodeNormal.h"


namespace fem_varden_flow

{
	/// Expression for a scalar far-field tracer at the ice-ocean interface.
	///
	/// The argument should be a dolfin function defined on the domain
	/// (mesh), typically a function representing the salinity or
	/// temperature field. The resulting Expression evaluates points
	/// on the boundary to be the "far-field" value measured at a
	/// distance `boundary_layer` in the normal direction away from
	/// the boundary (into the domain).
	/// See:

    /// - Holland and Jenkins (199): Modeling Thermodynamic Ice–Ocean
    ///   Interactions at the Base of an Ice Shelf.
    /// - Kimura et al. (2013): Adaptation of an unstructured-mesh,
    ///   ﬁnite-element ocean model to the simulation of ocean circulation
    ///   beneath ice shelves.
    /// - http://lxr.mitgcm.org/lxr2/source/MITgcm/pkg/shelfice/shelfice_thermodynamics.F
    ///   Lines: 466-467
	class FarFieldTracer : public dolfin::Expression
	{
	public:
		/// Constructor
		FarFieldTracer(std::shared_ptr<dolfin::Function> tracer,
					   std::shared_ptr<dolfin::Function> mesh_size,
					   std::shared_ptr<dolfin::Function> residual_indicator,
					   std::shared_ptr<dolfin::Function> u_x,
					   std::shared_ptr<dolfin::Function> u_y,
					   std::shared_ptr<dolfin::Function> u_z,
					   fem_varden_flow::NodeNormal& normal,
					   const dolfin::Parameters& p,
					   double zero_normal_tolerance=1e-12,
					   double kappa=1.0);

		/// Constructor
		FarFieldTracer(std::shared_ptr<dolfin::Function> tracer,
					   fem_varden_flow::NodeNormal& normal,
					   const dolfin::Parameters& p,
					   double zero_normal_tolerance=1e-12);

		/// Destructor
		~FarFieldTracer();

		/// Evaluates the farfield tracer at point x, based on normal
		/// direction and boundary-layer thickness
		virtual void eval(dolfin::Array<double>& values,
						  const dolfin::Array<double>& x) const override;

		void eval(dolfin::Array<double>& eval_values, dolfin::Array<double>& normal_values,
						  const dolfin::Array<double>& x) const;
		
		/// Return the default parameters (as defined by
		/// fem_varden_flow::ConfigurableParameters)
		static dolfin::Parameters default_parameters()
			{
				return fem_varden_flow::ConfigurableParameters::default_category_parameters("ocean_melt_model");
			}

		/// Parameters describing the far-field sampling procedure
		dolfin::Parameters parameters;
		
		/// Updates the parameter set
		void update_parameters(const dolfin::Parameters& parameters);
		
		/// Updates the parameter set
		void update_bl_thickness(double bl_thick)
			{
				bl_thick_ = bl_thick;
			}

		/// Field that will be evaluated at boundary layer thicknes away
		/// from mesh boundary
		std::shared_ptr<dolfin::Function> tracer_variable;
		
		std::shared_ptr<dolfin::Function> mesh_size_;
		std::shared_ptr<dolfin::Function> residual_indicator_;
		std::shared_ptr<dolfin::Function> u_x_;
		std::shared_ptr<dolfin::Function> u_y_;
		std::shared_ptr<dolfin::Function> u_z_;
		double kappa_;

		/// Specified boundary normal
		// should this be a shared pointer or just function
		// std::shared_ptr<dolfin::Function> boundary_normal;
		dolfin::Function boundary_normal;
		
	private:
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
		
		// Variables that 
		double bl_thick_;
		double n_tol_;
	};
}

#endif
