// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2022-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>

/* #include <dolfin.h> */
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>
#include <ConfigurableParameters.h>
#include <dolfin/mesh/CellType.h>
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
double xmin = -1.0 * H;
double xmax = 17.0 * H;
double ymin = 0.;
double ymax = 2*H;
double zmin = ymin;
double zmax = ymax;
double eps_ = 1e-8;
double x0 = 0.;

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

/// Analytic solutions
class IndataVelocity : public Expression {
public:

  double noise = 0.05; // Not sure exactly what should be here...

  IndataVelocity() : Expression(2) {}

  void eval(Array<double> &values, const Array<double> &x) const {
  float random_float;
    
    random_float = Random();
    values[0] = random_float * noise * exp(-25. * pow( x[0]  - 0.0 ,2));

    random_float = Random();
    values[1] = random_float * noise * exp(-25. * pow( x[0]  - 0.0 ,2));

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

/// Analytic solutions
class IndataVelocity3D : public Expression {
public:

  double noise = 0.05; // Not sure exactly what should be here...

  IndataVelocity3D() : Expression(3) {}

  void eval(Array<double> &values, const Array<double> &x) const {
  float random_float;
    
    random_float = Random();
    values[0] = random_float * noise * exp(-25. * pow( x[0]  - 0.0 ,2));

    random_float = Random();
    values[1] = random_float * noise * exp(-25. * pow( x[0]  - 0.0 ,2));

    random_float = Random();
    values[2] = random_float * noise * exp(-25. * pow( x[0]  - 0.0 ,2));

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

// Define top domain
class TopWall : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && x[1] > ymax - eps_;
  }
};

// Define bottom domain
class BottomWall : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && x[1] < ymin + eps_;
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


// Sub domain for Periodic boundary condition for x and z walls
class PeriodicBoundary : public SubDomain
{

  double eps = eps_;


  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return
      (
        (x[2] < zmin + eps)
       )
      // and
      // !(
        // ( (x[2] < zmin + eps))
        // ||
        // ( (x[2] > zmax - eps))
        // )
      and
      on_boundary;
  }

/*   // Check if a point is on the "positive" side of the boundary
  bool inside(const std::vector<double>& x, bool on_boundary) const
  {
    return std::abs(x[2] - zmax) < DOLFIN_EPS && on_boundary;
  }*/


  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
    if ( (x[2] > zmax - eps))
      {
        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2] - (zmax - zmin);
      }
    else if (x[2] > zmax - eps)
      {
        y[0] = x[0];
        y[1] = x[1];
        y[2] = x[2] - (zmax - zmin);
      }
    else
      {
        y[0] = -1000;
        y[1] = -1000;
        y[2] = -1000;
      }
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
