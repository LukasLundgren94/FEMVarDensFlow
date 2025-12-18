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

using namespace dolfin;

double d = 1.0;
double b = 0.01;

double xmin = 0.0;
double xmax = 0.0;
double ymin = 0.0;
double ymax = 0.0;
double rho_max = 0.0;
double eps_ = 1e-8;
double shift_rho = 2.;

/// Analytic solutions
class AnalyticVelocity : public Expression {
public:
  double &t;
  double hmin;
  int h_power = 1;

  AnalyticVelocity(double &t) : Expression(2), t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {
    // srand();
    // float random_float = (float)rand() / RAND_MAX;
    // int minus = rand() % 2;
    // minus = minus * (-1) + (1 - minus) * 1;

    // random_float = random_float * minus;
    // float random_float = Random();
    // std::cout << "rand: " << random_float << std::endl;
    double u_max_l2 = sqrt(2*M_PI*2*M_PI + 2*M_PI*2*M_PI);

    values[0] = -x[1] * M_PI * 2.0;// + u_max_l2*Random() *  pow(hmin,h_power);



    // random_float = random_float * minus;
    values[1] = x[0] * M_PI * 2.0;// +  u_max_l2*Random() * pow(hmin,h_power);
  }

private:
  float Random() const {
  float random_float = (float)rand() / RAND_MAX;
    int minus = rand() % 2;
    minus = minus * (-1) + (1 - minus) * 1;

    random_float = random_float * minus;
    return random_float;
  }

};

class AnalyticDensity : public Expression {
public:
  double &t;

  AnalyticDensity(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    double x0 = 0.35 * cos(t * 2 * M_PI);
    double y0 = 0.35 * sin(t * 2 * M_PI);
    double r0 = 0.25;
    values[0] = shift_rho + 
        0.5 *
        (1 - tanh(((x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0)) /
                      (r0 * r0) -
                  1));
  }
};

class AnalyticDensityCylinder : public Expression {
public:
  double &t;

  AnalyticDensityCylinder(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    double x0 = 0.35 * cos(t * 2 * M_PI);
    double y0 = 0.35 * sin(t * 2 * M_PI);
    double r0 = 0.25;

    double r = (x[0] - x0) * (x[0] - x0) + (x[1] - y0) * (x[1] - y0);
    if (r <= r0 * r0) {
      values[0] = 1.0;
    } else {
      values[0] = 0.0;
    }
  }
};

class AnalyticPressure : public Expression {
public:
  double &t;

  AnalyticPressure(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {

    values[0] = 1.00;
  }
};

class CompleteBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary;
  }
};

class InflowBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && (x[1] < ymin + eps_ || x[0] < xmin + eps_);
  }
};

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

class SlipBC_marker : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 1.0;
  }
};

class SlipBC_marker_fn : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 1.0;
  }
};