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


double H = 1.0;
double gamma_ = 2.0;
double rho1 = 1.0;
double rho2 = 0.7;
double g = 1.0;
double xmin = 0.0 * H;
double xmax = 32.0 * H;
double ymin = -H / 2.;
double ymax = H / 2.;
double zmin = ymin;
double zmax = ymax;
double eps_ = 1e-8;
double x0 = 14.;

/// Initial condition density
class IndataDensity : public Expression {
public:
  double &t;
  double Re;
  Parameters solver_parameters;

  IndataDensity(double &t) : t(t) {
    const auto &param = shared_solver_parameters();
    solver_parameters = param;
    Re = solver_parameters["nu"];
  }

  void eval(Array<double> &values, const Array<double> &x) const {
    double Sc = 1.;
    double xp = x[0] - x0;
    values[0] = 0.5 * (rho2 / rho1 + 1.) -
                0.5 * (1. - rho2 / rho1) * erf(xp * sqrt(Re * Sc));
  }
};

// Define noslip domain
class NoslipDomain : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && (x[1] < ymin + eps_ || x[1] > ymax - eps_);
  }
};

// Define noslip domain
class SlipDomain : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && (x[0] < xmin + eps_ || x[0] > xmax - eps_);
  }
};

//

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

/// Local refinement
void local_refinement(Mesh &mesh) {
  for (int k = 1; k < 1; ++k) {
    MeshFunction<bool> cell_markers(std::shared_ptr<Mesh>(&mesh, NoDeleter()),
                                    mesh.topology().dim(), false);
    for (CellIterator c(mesh); !c.end(); ++c) {
      if (c->midpoint().y() < 0.3 / k || c->midpoint().y() > 2.0 - 0.3 / k)
        cell_markers[*c] = true;
    }
    mesh = refine(mesh, cell_markers);
  }
};

class CompleteBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary;
  }
};

class SlipBC_marker : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 0.0;

    /*    if (((x[0] > xmin  && x[0] < xmax -0.1  - eps_) &&
            (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmin  &&
       x[0] < xmax  - eps_ - 0.1) ) { values[0] = 1.0;
        }*/
  }
};

class SlipBC_marker_fn : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 1.0;

    /*    if (((x[0] > xmax - 0.1  && x[0] < xmax  - eps_) &&
            (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmax - 0.1
       && x[0] < xmax  - eps_ ) ) { values[0] = 1.0;
        }*/
  }
};
