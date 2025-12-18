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


double xmin = -0.5;
double xmax = 0.5;
double ymin = -0.5;
double ymax = 0.5;
double zmin = ymin;
double zmax = ymax;
double eps_ = 1e-8;
double radius = 1.0;

#define CONSTANT_DENSITY false

/// Analytic solutions
class AnalyticVelocity : public Expression {
public:
  double &t;

  AnalyticVelocity(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

        const auto &param = shared_solver_parameters();
    double nu = param["nu"];
    double rho;
#if CONSTANT_DENSITY
    rho = 1.0;
#else
    double x0 = 1.0; // 0.35 * cos(t * 2 * M_PI);
    double y0 = 1.0; // 0.35 * sin(t * 2 * M_PI);
    double r0 = 0.05;
    rho = 1. + 0.5 * (1 - tanh(((x[0] - x0) * (x[0] - x0) +
                                (x[1] - y0) * (x[1] - y0)) /
                                   (r0 * r0) -
                               1));
#endif
    nu /= rho;
    double const pi = DOLFIN_PI;
    values[0] =  -sin(2*pi * (x[1] - .5)) * pow(sin(pi * (x[0] - .5)),2)*exp(-2.0 * t * nu * pi * pi);
    values[1] = sin(2*pi *(x[0] - .5) ) * pow(sin(pi * (x[1] - .5)  ),2)*exp(-2.0 * t * nu * pi * pi);
  }
};

class AnalyticMomentum : public Expression {
public:
  double &t;

  AnalyticMomentum(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

        const auto &param = shared_solver_parameters();
    double nu = param["nu"];
    double rho;
#if CONSTANT_DENSITY
    rho = 1.0;
#else
    double x0 = 1.0; // 0.35 * cos(t * 2 * M_PI);
    double y0 = 1.0; // 0.35 * sin(t * 2 * M_PI);
    double r0 = 0.05;
    rho = 1. + 0.5 * (1 - tanh(((x[0] - x0) * (x[0] - x0) +
                                (x[1] - y0) * (x[1] - y0)) /
                                   (r0 * r0) -
                               1));
#endif
    nu /= rho;
    double const pi = DOLFIN_PI;
    values[0] =  -rho*sin(2*pi * (x[1] - .5)) * pow(sin(pi * (x[0] - .5)),2)*exp(-2.0 * t * nu * pi * pi);
    values[1] = rho*sin(2*pi *(x[0] - .5) ) * pow(sin(pi * (x[1] - .5)  ),2)*exp(-2.0 * t * nu * pi * pi);
  }
};

class AnalyticVelocity_t : public Expression {
public:
  double &t;

  AnalyticVelocity_t(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = x[1] * sin(t);
    values[1] = -x[0] * sin(t);
  }
};

class AnalyticVelocity_tt : public Expression {
public:
  double &t;

  AnalyticVelocity_tt(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = x[1] * cos(t);
    values[1] = -x[0] * cos(t);
  }
};

class AnalyticVelocity_ttt : public Expression {
public:
  double &t;

  AnalyticVelocity_ttt(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = -x[1] * sin(t);
    values[1] = x[0] * sin(t);
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

    double x0 = (0.5); // 0.35 * cos(t * 2 * M_PI);
    double y0 = 0.5;   // 0.35 * sin(t * 2 * M_PI);

    double r0 = 0.2;
    double r = sqrt((x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0));

    values[0] = 1. + 0.25 * (tanh(10 * (r - r0)));
    
/*    double const speed = 1.0;
    double x0 = 0. * cos(t * 2 * M_PI);
    double y0 = 0. * sin(t * 2 * M_PI);
    double r0 = 0.2;
    values[0] = 1. + 0.5 * (1 - tanh(((x[0] - x0) * (x[0] - x0) +
                                      (x[1] - y0) * (x[1] - y0)) /
                                         (r0 * r0) -
                                     1));*/
#endif
  }
};

class AnalyticDensity_t : public Expression {
public:
  double &t;

  AnalyticDensity_t(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

#if CONSTANT_DENSITY
    values[0] = 0.0;
#else
    values[0] = cos(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t)));
#endif
  }
};

class AnalyticDensity_tt : public Expression {
public:
  double &t;

  AnalyticDensity_tt(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

#if CONSTANT_DENSITY
    values[0] = 0.0;
#else
    values[0] =
        -cos(t) * (x[1] * sin(sin(t)) * cos(t) + x[0] * cos(sin(t)) * cos(t)) -
        sin(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t)));
#endif
  }
};

class AnalyticDensity_ttt : public Expression {
public:
  double &t;

  AnalyticDensity_ttt(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

#if CONSTANT_DENSITY
    values[0] = 0.0;
#else
    values[0] = cos(t) * (-x[1] * cos(sin(t)) + x[0] * sin(sin(t)) +
                          x[0] * cos(sin(t)) * sin(t) * 3.0 +
                          x[1] * sin(sin(t)) * sin(t) * 3.0 -
                          x[1] * cos(sin(t)) * pow(cos(t), 2.0) +
                          x[0] * sin(sin(t)) * pow(cos(t), 2.0));
#endif
  }
};

class AnalyticPressure : public Expression {
public:
  double &t;

  AnalyticPressure(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

        const auto &param = shared_solver_parameters();
    double const nu = param["nu"];
    double const speed = 1.0;
    double const pi = DOLFIN_PI;
    double const p = -0.25 * (cos(2.0 * pi * (x[0] - speed*t)) + cos(2.0 * pi * x[1])) *
                     exp(-4.0 * t * nu * pi * pi);
    values[0] = p;
  }
};

class AnalyticPressure_t : public Expression {
public:
  double &t;

  AnalyticPressure_t(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = sin(x[0]) * sin(x[1]) * cos(t);
  }
};

class AnalyticPressure_tt : public Expression {
public:
  double &t;

  AnalyticPressure_tt(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = -sin(x[0]) * sin(x[1]) * sin(t);
  }
};

class AnalyticPressure_ttt : public Expression {
public:
  double &t;

  AnalyticPressure_ttt(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = -sin(x[0]) * sin(x[1]) * cos(t);
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

class ForcingFunction_t : public Expression {
public:
  double &t;

  ForcingFunction_t(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

#if CONSTANT_DENSITY
    values[0] = x[1] * cos(t) + 2 * x[0] * cos(t) * sin(t) +
                cos(t) * cos(x[0]) * sin(x[1]);
    values[1] = 2 * x[1] * cos(t) * sin(t) - x[0] * cos(t) +
                cos(t) * cos(x[1]) * sin(x[0]);
#else
    values[0] = (x[1] * cos(t) + x[0] * cos(t) * sin(t) * 2.0) *
                    (x[0] * cos(sin(t)) + x[1] * sin(sin(t)) + 2.0) +
                cos(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) *
                    (-x[0] + x[0] * pow(sin(t), 2.0) + x[1] * sin(t)) +
                cos(t) * cos(x[0]) * sin(x[1]);
    values[1] = -(x[0] * cos(t) - x[1] * cos(t) * sin(t) * 2.0) *
                    (x[0] * cos(sin(t)) + x[1] * sin(sin(t)) + 2.0) -
                cos(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) *
                    (x[1] - x[1] * pow(sin(t), 2.0) + x[0] * sin(t)) +
                cos(t) * cos(x[1]) * sin(x[0]);
#endif
  }
};

class ForcingFunction_tt : public Expression {
public:
  double &t;

  ForcingFunction_tt(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

#if CONSTANT_DENSITY
    values[0] = 2 * x[0] * cos(t) * cos(t) - 2 * x[0] * sin(t) * sin(t) -
                x[1] * sin(t) - cos(x[0]) * sin(t) * sin(x[1]);
    values[1] = 2 * x[1] * cos(t) * cos(t) - 2 * x[1] * sin(t) * sin(t) +
                x[0] * sin(t) - cos(x[1]) * sin(t) * sin(x[0]);
#else
    values[0] = -(x[0] * -2.0 + x[0] * pow(sin(t), 2.0) * 4.0 + x[1] * sin(t)) *
                    (x[0] * cos(sin(t)) + x[1] * sin(sin(t)) + 2.0) -
                cos(x[0]) * sin(t) * sin(x[1]) -
                sin(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) *
                    (-x[0] + x[0] * pow(sin(t), 2.0) + x[1] * sin(t)) -
                pow(cos(t), 2.0) * (x[0] * cos(sin(t)) + x[1] * sin(sin(t))) *
                    (-x[0] + x[0] * pow(sin(t), 2.0) + x[1] * sin(t)) +
                cos(t) * (x[1] * cos(t) + x[0] * cos(t) * sin(t) * 2.0) *
                    (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) * 2.0;

    values[1] = (x[1] * 2.0 - x[1] * pow(sin(t), 2.0) * 4.0 + x[0] * sin(t)) *
                    (x[0] * cos(sin(t)) + x[1] * sin(sin(t)) + 2.0) +
                pow(cos(t), 2.0) * (x[0] * cos(sin(t)) + x[1] * sin(sin(t))) *
                    (x[1] - x[1] * pow(sin(t), 2.0) + x[0] * sin(t)) -
                cos(x[1]) * sin(t) * sin(x[0]) -
                cos(t) * (x[0] * cos(t) - x[1] * cos(t) * sin(t) * 2.0) *
                    (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) * 2.0 +
                sin(t) * (x[1] * cos(sin(t)) - x[0] * sin(sin(t))) *
                    (x[1] - x[1] * pow(sin(t), 2.0) + x[0] * sin(t));
#endif
  }
};

class ForcingFunction_ttt : public Expression {
public:
  double &t;

  ForcingFunction_ttt(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {
    double xx = x[0];
    double y = x[1];

#if CONSTANT_DENSITY
    values[0] = -cos(t) * (y + cos(xx) * sin(y) + 8 * xx * sin(t));
    values[1] = -cos(t) * (cos(y) * sin(xx) - xx + 8 * y * sin(t));
#else
    values[0] = -cos(t) *
                (x[1] * 2.0 + cos(x[0]) * sin(x[1]) -
                 (x[0] * x[0]) * sin(sin(t)) * 1.2E+1 -
                 (x[1] * x[1]) * sin(sin(t)) * 2.0 + x[0] * sin(t) * 1.6E+1 +
                 (x[0] * x[0]) * cos(sin(t)) * sin(t) * 8.0 +
                 (x[1] * x[1]) * cos(sin(t)) * sin(t) * 7.0 +
                 x[0] * x[1] * cos(sin(t)) * 1.0E+1 +
                 (x[0] * x[0]) * sin(sin(t)) * pow(cos(t), 2.0) * 1.9E+1 +
                 (x[0] * x[0]) * sin(sin(t)) * pow(cos(t), 4.0) +
                 (x[1] * x[1]) * sin(sin(t)) * pow(cos(t), 2.0) * 6.0 +
                 x[0] * x[1] * sin(sin(t)) * sin(t) +
                 (x[0] * x[0]) * cos(sin(t)) * pow(cos(t), 2.0) * sin(t) * 9.0 +
                 (x[1] * x[1]) * cos(sin(t)) * pow(cos(t), 2.0) * sin(t) -
                 x[0] * x[1] * cos(sin(t)) * pow(cos(t), 2.0) * 1.3E+1 -
                 x[0] * x[1] * cos(sin(t)) * pow(cos(t), 4.0) +
                 x[0] * x[1] * sin(sin(t)) * pow(cos(t), 2.0) * sin(t) * 8.0);
    values[1] = -cos(t) *
                (x[0] * -2.0 + cos(x[1]) * sin(x[0]) +
                 (x[0] * x[0]) * cos(sin(t)) * 2.0 +
                 (x[1] * x[1]) * cos(sin(t)) * 1.2E+1 + x[1] * sin(t) * 1.6E+1 +
                 (x[0] * x[0]) * sin(sin(t)) * sin(t) * 7.0 +
                 (x[1] * x[1]) * sin(sin(t)) * sin(t) * 8.0 -
                 x[0] * x[1] * sin(sin(t)) * 1.0E+1 -
                 (x[0] * x[0]) * cos(sin(t)) * pow(cos(t), 2.0) * 6.0 -
                 (x[1] * x[1]) * cos(sin(t)) * pow(cos(t), 2.0) * 1.9E+1 -
                 (x[1] * x[1]) * cos(sin(t)) * pow(cos(t), 4.0) +
                 x[0] * x[1] * cos(sin(t)) * sin(t) +
                 (x[0] * x[0]) * sin(sin(t)) * pow(cos(t), 2.0) * sin(t) +
                 (x[1] * x[1]) * sin(sin(t)) * pow(cos(t), 2.0) * sin(t) * 9.0 +
                 x[0] * x[1] * sin(sin(t)) * pow(cos(t), 2.0) * 1.3E+1 +
                 x[0] * x[1] * sin(sin(t)) * pow(cos(t), 4.0) +
                 x[0] * x[1] * cos(sin(t)) * pow(cos(t), 2.0) * sin(t) * 8.0);

#endif
  }
};

// Define noslip domain
class CirclePoint : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return (x[1] > ymax - eps_);
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
//Sub domain for Periodic boundary condition
class PeriodicBoundaryXY1: public SubDomain
{
  double const eps = 1.0e-8;
 /* double xmin = 0.0;
  double xmax = 1.0; //2.0 * DOLFIN_PI;
  double ymin = 0.0;
  double ymax = 1.0; //2.0 * DOLFIN_PI;
*/
  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return
      (
       (x[0] < xmin + eps) || (x[1] < ymin + eps)
       )
      &&
      !(
    ((x[0] > xmax - eps) && (x[1] < ymin + eps))
    ||
    ((x[0] < xmin + eps) && (x[1] > ymax - eps))
    )
      &&
      on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {

    if ((x[0] > xmax - eps) && (x[1] > ymax - eps))
      {
    y[0] = x[0] - (xmax - xmin);
    y[1] = x[1] - (ymax - ymin);
      }
    else if (x[0] > xmax - eps)
      {
    y[0] = x[0] - (xmax - xmin);
    y[1] = x[1];
      }
    else if (x[1] > ymax - eps)
      {
    y[0] = x[0];
    y[1] = x[1] - (ymax - ymin);
      }
    else
      {
    y[0] = -1000.0;
    y[1] = -1000.0;
      }
  }
};