// Copyright (C) 2023 Christian Helanow
// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
//
// First added:  2023-01-07

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
// #include <dolfin/mesh/Mesh.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>

// #include <dolfin/function/GenericFunction.h>
// #include <dolfin/function/Expression.h>
// #include <dolfin/common/Array.h>
// #include <dolfin/mesh/Cell.h>
// #include <dolfin/mesh/Mesh.h>
// #include <dolfin/function/FunctionSpace.h>

// general
#include <cmath>

// fem_varden_flow
#include "NodeNormal.h"
#include "FarFieldTracer.h"

using namespace fem_varden_flow;

FarFieldTracer::FarFieldTracer(
    std::shared_ptr<dolfin::Function> tracer,
    std::shared_ptr<dolfin::Function> mesh_size,
    std::shared_ptr<dolfin::Function> residual_indicator,
    std::shared_ptr<dolfin::Function> u_x,
    std::shared_ptr<dolfin::Function> u_y,
    std::shared_ptr<dolfin::Function> u_z, fem_varden_flow::NodeNormal &normal,
    const dolfin::Parameters &p, double zero_normal_tolerance, double kappa)
    : FarFieldTracer(tracer, normal, p, zero_normal_tolerance) {
  kappa_ = kappa;
  mesh_size_ = mesh_size;
  residual_indicator_ = residual_indicator;
  u_x_ = u_x;
  u_y_ = u_y;
  u_z_ = u_z;
}

FarFieldTracer::FarFieldTracer(std::shared_ptr<dolfin::Function> tracer,
                               fem_varden_flow::NodeNormal &normal,
                               const dolfin::Parameters &p,
                               double zero_normal_tolerance)
    : dolfin::Expression() {
  if (!(tracer->value_rank() == 0 && tracer->value_dimension(0) == 1)) {
    dolfin::error("tracer function must be a scalar function.");
  }
  tracer_variable = tracer;
  tracer_variable->set_allow_extrapolation(true);
  update_parameters(p);
  n_tol_ = zero_normal_tolerance;
  boundary_normal = normal.normal();
  boundary_normal.set_allow_extrapolation(true);
}

FarFieldTracer::~FarFieldTracer() {
  // Do nothing
}

void FarFieldTracer::update_parameters(const dolfin::Parameters& input)
{
  this->parameters = input;
  bl_thick_ = this->parameters["boundary_layer_thickness"];
}

void FarFieldTracer::eval(dolfin::Array<double> &values,
                          const dolfin::Array<double> &x) const {
  // if magnitude of normal is not (near) 1.0, the point should not
  // be on any part of the boundary for which the far-field
  // variable is to be evaluated, so return zero
  // Note: this is based on the assumption that the normal is a
  // CG1 function and normalized everywhyere on the boundary: any
  // value less/other than |n| = 1 should indicate that the
  // normal is evaluated away from the boundary.

  // CH: should I assert sizes here?

  dolfin::Array<double> normal_pointval(boundary_normal.value_size());
  try {
    boundary_normal.eval(normal_pointval, x);

  } catch (std::exception &e) {
    dolfin::warning("FarFieldTracer: Error in boundary_normal.eval().");
    array_fill(values, NAN);
    return;
  }
  // test_val(normal_pointval, "normal_pointval");

  if (dolfin::near(array_norm(normal_pointval), 0.0, n_tol_)) {
    array_fill(values, NAN);
    return;
  }

  dolfin::Array<double> scaled_distance(normal_pointval.size());
  //   if (mesh_size_ && residual_indicator_ && u_x_) {
  //     dolfin::Array<double> res_pointval(1);
  //     dolfin::Array<double> u_x_pointval(1);
  //     dolfin::Array<double> u_y_pointval(1);
  //     dolfin::Array<double> mesh_pointval(normal_pointval.size());
  //     residual_indicator_->eval(res_pointval, x);
  //     u_x_->eval(u_x_pointval, x);
  //     u_y_->eval(u_y_pointval, x);
  //     mesh_size_->eval(mesh_pointval, x);

  //     scaled_distance[0] = kappa_ / mesh_pointval[0] +
  //                          res_pointval[0] * mesh_pointval[0] *
  //                          u_x_pointval[0];
  //     scaled_distance[1] = kappa_ / mesh_pointval[1] +
  //                          res_pointval[0] * mesh_pointval[1] *
  //                          u_y_pointval[0];

  //     if (normal_pointval.size() == 3) {
  //       dolfin::Array<double> u_z_pointval(1);
  //       u_z_->eval(u_z_pointval, x);
  //       scaled_distance[2] = res_pointval[0] * mesh_pointval[2] *
  //       u_z_pointval[0];
  //     }

  //     double vloc = sqrt(u_x_pointval[0] * u_x_pointval[0] +
  //                        u_y_pointval[0] * u_y_pointval[0]);

  //     double h = sqrt(mesh_pointval[0] * mesh_pointval[0] +
  //                     mesh_pointval[1] * mesh_pointval[1]);
  //     scaled_distance[0] = res_pointval[0] * h * vloc + kappa_ / h;
  //     scaled_distance[1] = res_pointval[0] * h * vloc + kappa_ / h;

  //     // std::cout << "FarFieldTracer: scaled_distance = " <<
  //     scaled_distance[0]
  //     // << std::endl;
  //   } else {
  for (std::size_t i = 0; i < normal_pointval.size(); ++i) {
    scaled_distance[i] = bl_thick_;
  }
  //   }

  // get value at far-field position: outward-pointing boundary
  // normal, so minus is for into the domain
  dolfin::Array<double> farfield_pos(normal_pointval.size());

  for (std::size_t i = 0; i < farfield_pos.size(); ++i) {
    farfield_pos[i] = x[i] - scaled_distance[i] * normal_pointval[i];
  }
  // test_val(x, "x");
  // test_val(farfield_pos, "farfield_pos");

  // eval the farfield variable at the position
  try {
    tracer_variable->eval(values, farfield_pos);
  } catch (std::exception &e) {
    dolfin::warning("FarFieldTracer: Error in tracer_variable.eval().");
    array_fill(values, NAN);
  }
  // test_val(values, "values");
}

void FarFieldTracer::eval(dolfin::Array<double> &eval_values,
                          dolfin::Array<double> &normal_values,
                          const dolfin::Array<double> &x) const {

  dolfin::Array<double> eval_value_scalar(1);

  // dolfin::Array<double> normal_pointval(boundary_normal.value_size());
  try {
    boundary_normal.eval(normal_values, x);

  } catch (std::exception &e) {
    dolfin::warning("FarFieldTracer: Error in boundary_normal.eval().");
    array_fill(eval_values, NAN);
    return;
  }
  // test_val(normal_pointval, "normal_pointval");

  if (dolfin::near(array_norm(normal_values), 0.0, n_tol_)) {
    array_fill(eval_values, NAN);
    return;
  }

  dolfin::Array<double> scaled_distance(normal_values.size());

  for (std::size_t i = 0; i < normal_values.size(); ++i) {
    scaled_distance[i] = bl_thick_;
  }

  // get value at far-field position: outward-pointing boundary
  // normal, so minus is for into the domain
  dolfin::Array<double> farfield_pos(normal_values.size());

  for (std::size_t i = 0; i < farfield_pos.size(); ++i) {

    for (std::size_t j = 0; j < farfield_pos.size(); ++j) {
      if (j == i) {
        
        farfield_pos[j] = x[j] - scaled_distance[j] * normal_values[j] /
                                     std::abs(normal_values[j]);
      } else {
        farfield_pos[j] = x[j];
      }
    }

    try {
      tracer_variable->eval(eval_value_scalar, farfield_pos);
    } catch (std::exception &e) {
      dolfin::warning("FarFieldTracer: Error in tracer_variable.eval().");
      array_fill(eval_value_scalar, NAN);
    }
    eval_values[i] = eval_value_scalar[0];
  }
}

// CH: cell version. I guess this can be avoided since the
// dolfin::Functions are used for evaluation and can just use the
// point (x). However, not sure if this would be a big computational
// penalty

// void FarFieldTracer::eval(Array<double>& values, const Array<double>& x,
// 						  const ufc::cell& cell) const
// {
// 	// if magnitude of normal is not (near) 1.0, the point should not
// 	// be on any part of the boundary for which the far-field
// 	// variable is to be evaluated, so return zero
// 	// Note: this is based on the assumption that the normal is a
// 	// CG1 function and normalized everywhyere on the boundary: any
// 	// value less/other than |n| = 1 should indicate that the
// 	// normal is evaluated away from the boundary.

// 	// CH: should I assert sizes here?

// 	dolfin::Array<double> normal_pointval(boundary_normal.value_size());
// 	boundary_normal.eval(normal_pointval, x, cell);
// 	if (dolfin::near(array_norm(normal_pointval), 0.0, n_tol_))
// 	{
// 		// array_fill(values, 0.0);
// 		values[0] = 0;
// 		return;
// 	}
// 	// get value at far-field position: outward-pointing boundary
// 	// normal, so minus is for into the domain
// 	dolfin::Array<double> farfield_pos(normal_pointval.size());

// 	for (std::size_t i = 0; i < farfield_pos.size(); ++i)
// 	{
// 		farfield_pos[i] = x[i] - bl_thick_ * normal_pointval[i];
// 	}
// 	// eval the farifield variable at the position
// 	dolfin::Array<double> farfield_pointval(1);
// 	tracer_variable->eval(farfield_pointval, x, cell);
// }
