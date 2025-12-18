// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
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



double xmin = -1.0;
double xmax = 1.0;
double ymin = -1.0;
double ymax = 1.0;
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

    values[0] = -x[1] * cos(t);
    values[1] = x[0] * cos(t);
  }
};

class AnalyticMomentum : public Expression {
public:
  double &t;

  AnalyticMomentum(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    double rho = 2.0 + x[0] * cos(sin(t)) + x[1] * sin(sin(t));

    values[0] = -x[1] * cos(t) * rho;
    values[1] = x[0] * cos(t) * rho;
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
     values[0] = 2.0 + x[0] * cos(sin(t)) + x[1] * sin(sin(t));
 #endif
  }
};

class AnalyticPressure : public Expression {
public:
  double &t;

  double normalization_value = 0.0;
  double rho_bar = 0.0;

  AnalyticPressure(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

        const auto &param = shared_solver_parameters();

    double const emac_shift = param["emac_pressure_shift"];
    double const skew_rho = param["skew_rho_factor"];
    int const remove_rho_average = param["remove_rho_average"];
      double const u0 = -x[1] * cos(t);
    double const u1  = x[0] * cos(t);
      double const uu = u0*u0 + u1*u1;
      double const rho = 2.0 + x[0] * cos(sin(t)) + x[1] * sin(sin(t)); //cos(t)*(x[1]*cos(sin(t))-x[0]*sin(sin(t)));
      double shift = uu*(rho);//-rho_bar*remove_rho_average*skew_rho);
      double shift_bar = 0.5*uu*((rho_bar)*remove_rho_average*skew_rho);

    values[0] = sin(x[0]) * sin(x[1]) * sin(t) - emac_shift*shift - shift_bar  - normalization_value;
  }
};

class ForcingFunction : public Expression {
public:
  double &t;

  ForcingFunction(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

            double x0 = 0.35 * cos(sin(t)*3);
    double y0 = 0.35 * sin(sin(t)*3);
    double r0 = 0.25;
    double shift_rho = 1.0;
   double rho = shift_rho + 
        0.5 *
        (1 - tanh(((x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0)) /
                      (r0 * r0) -
                  1));


#if CONSTANT_DENSITY
    double rho = 1.0;
    values[0] = rho * (x[1] * sin(t) - x[0] * cos(t) * cos(t)) +
                cos(x[0]) * sin(x[1]) * sin(t);
    values[1] = -rho * (x[0] * sin(t) + x[1] * cos(t) * cos(t)) +
                sin(x[0]) * cos(x[1]) * sin(t);
#else
    // velocity formulation
    // values[0] = (-x[0]+x[0]*pow(sin(t),2.0)+x[1]*sin(t))*(x[0]*cos(sin(t))+x[1]*sin(sin(t))+2.0)+cos(x[0])*sin(t)*sin(x[1]);
    // values[1] = -(x[0]*cos(sin(t))+x[1]*sin(sin(t))+2.0)*(x[1]-x[1]*pow(sin(t),2.0)+x[0]*sin(t))+cos(x[1])*sin(t)*sin(x[0]);

    // momentum formulation (same)
    values[0] =       x[0]*pow(cos(t),2.0)*-2.0+x[1]*sin(t)*2.0+cos(x[0])*sin(t)*sin(x[1])+(x[1]*x[1])*sin(sin(t))*sin(t)-(x[0]*x[0])*cos(sin(t))*pow(cos(t),2.0)+x[0]*x[1]*cos(sin(t))*sin(t)-x[0]*x[1]*sin(sin(t))*pow(cos(t),2.0);
    values[1] =     x[1]*pow(cos(t),2.0)*-2.0-x[0]*sin(t)*2.0+cos(x[1])*sin(t)*sin(x[0])-(x[0]*x[0])*cos(sin(t))*sin(t)-(x[1]*x[1])*sin(sin(t))*pow(cos(t),2.0)-x[0]*x[1]*sin(sin(t))*sin(t)-x[0]*x[1]*cos(sin(t))*pow(cos(t),2.0);




#endif
    
  }
};

class CompleteBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary;
  }
};

class SlipBC_marker : public Expression {
  CompleteBoundary cb;

  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 1.0;
  }
};

class SlipBC_marker_fn : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 1.0;
  }
};