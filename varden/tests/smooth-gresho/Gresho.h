// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2022-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>

/* #include <dolfin.h> */
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>
#include <ConfigurableParameters.h>
#include <memory>

using namespace dolfin;

namespace {
inline const Parameters& shared_solver_parameters() {
  static const Parameters* params = []() {
    fem_varden_flow::ConfigurableParameters config("parameters.json");
    return new Parameters(config.to_dolfin_parameters("solver_parameters"));
  }();
  return *params;
}

} // namespace


double stretch_factor = 2.5;

double xmin = -0.5*stretch_factor;
double xmax = 0.5*stretch_factor;
double ymin = -0.5*stretch_factor;
double ymax = 0.5*stretch_factor;
double zmin = ymin;
double zmax = ymax;
double eps_ = 1e-8;
double radius = 0.125;

#define CONSTANT_DENSITY true

/// Analytic solutions
class AnalyticVelocity : public Expression {
public:
  double &t;

  AnalyticVelocity(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

        const auto &param = shared_solver_parameters();
    double nu = param["nu"];
    double rho;



    double const C2 = (-12.5) * (0.4 * 0.4) + 20 * (0.4 * 0.4) - 4 * log(0.4);
    double const C1 = C2 - 20 * 0.2 + 4 * log(0.2);

    // if (r <= 0.2) {

double x0 = (xmax + xmin) / 2.0; // 1.0; // 0.35 * cos(t * 2 * M_PI);
    double y0 = (ymax + ymin) / 2.0; // 1.0; // 0.35 * sin(t * 2 * M_PI);
    double const r = sqrt(x[0] * x[0] + x[1] * x[1]);

    double r0 = radius;
    double compact_support = 0.1 * (1 - tanh(((x[0] - x0) * (x[0] - x0) +
                                       (x[1] - y0) * (x[1] - y0)) /
                                          (r0 * r0) -
                                      1));


      values[0] = - 5*x[1]*compact_support; //exp( pow(r,2) );
      values[1] =  5*x[0]*compact_support; //exp( pow(r,2) );
    // } else if (r <= 0.4) {
      // values[0] = -2 * x[1] / r + 5 * x[1];
      // values[1] = 2 * x[0] / r - 5 * x[0];
    // } else {
      // values[0] = 0.0;
      // values[1] = 0.0;
    // }
  }
};

class AnalyticDensity : public Expression {
public:
  double &t;

  AnalyticDensity(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {
#if CONSTANT_DENSITY
    values[0] = 1.0;
#else
    double x0 = (xmax + xmin) / 2.0; // 1.0; // 0.35 * cos(t * 2 * M_PI);
    double y0 = (ymax + ymin) / 2.0; // 1.0; // 0.35 * sin(t * 2 * M_PI);
    double r0 = radius;
    double rho = 1. + 0.5 * (1 - tanh(((x[0] - x0) * (x[0] - x0) +
                                       (x[1] - y0) * (x[1] - y0)) /
                                          (r0 * r0) -
                                      1));

    values[0] = rho;
#endif
  }
};

class AnalyticPressure : public Expression {
public:
  double &t;

  AnalyticPressure(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    double const r = sqrt(x[0] * x[0] + x[1] * x[1]);

    double const C2 = (-12.5) * (0.4 * 0.4) + 20 * (0.4 * 0.4) - 4 * log(0.4);
    double const C1 = C2 - 20 * 0.2 + 4 * log(0.2);

    if (r <= 0.2) {
      values[0] = 12.5 * r * r + C1;
    } else if (r <= 0.4) {
      values[0] = 12.5 * r * r - 20 * r + 4 * log(r) + C2;
    } else {
      values[0] = 0.0;
    }
  }
};

class ForcingFunction : public Expression {
public:
  double &t;

  ForcingFunction(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = 0.0;
    values[1] = 0.0;
  }
};

class CompleteBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary;
  }
};

// Sub domain for Periodic boundary condition for x and z walls
class PeriodicBoundary : public SubDomain {

  // double xmin = 0.0;
  // double xmax = 10.0;
  // double ymin = 0.0;
  // double ymax = 2.0;
  // double zmin = 0.0;
  // double zmax = 3.0;
  // double eps = DOLFIN_EPS_LARGE;
  double eps = eps_;

  // Left boundary is "target domain" G
  bool inside(const Array<double> &x, bool on_boundary) const {
    return ((x[0] < xmin + eps)) and on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double> &x, Array<double> &y) const {
    y[0] = x[0] - (xmax - xmin);
    y[1] = x[1];
  }
};

///
// Sub domain for Periodic boundary condition
class PeriodicBoundaryXY1 : public SubDomain {
  double const eps = 1.0e-8;
  /* double xmin = 0.0;
   double xmax = 1.0; //2.0 * DOLFIN_PI;
   double ymin = 0.0;
   double ymax = 1.0; //2.0 * DOLFIN_PI;
 */
  // Left boundary is "target domain" G
  bool inside(const Array<double> &x, bool on_boundary) const {
    return ((x[0] < xmin + eps) || (x[1] < ymin + eps)) &&
           !(((x[0] > xmax - eps) && (x[1] < ymin + eps)) ||
             ((x[0] < xmin + eps) && (x[1] > ymax - eps))) &&
           on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double> &x, Array<double> &y) const {

    if ((x[0] > xmax - eps) && (x[1] > ymax - eps)) {
      y[0] = x[0] - (xmax - xmin);
      y[1] = x[1] - (ymax - ymin);
    } else if (x[0] > xmax - eps) {
      y[0] = x[0] - (xmax - xmin);
      y[1] = x[1];
    } else if (x[1] > ymax - eps) {
      y[0] = x[0];
      y[1] = x[1] - (ymax - ymin);
    } else {
      y[0] = -1000.0;
      y[1] = -1000.0;
    }
  }
};


class TopBottom : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && ( (x[1] > ymax - eps_) || (x[1] < ymin + eps_) );
  }
};

class LeftRight : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && ( (x[0] > xmax - eps_) || (x[0] < xmin + eps_) );
  }
};