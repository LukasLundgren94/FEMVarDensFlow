// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2021-2025 Lukas Lundgren
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
double b = 0.1;
double rho1 = 1000.0;
double rho2 = 1.0;
double g = 1.0;
double xmin = 0;
double xmax = d/2.0;
double ymin = 0;
double ymax = 2.0 * d;
double zmin = xmin;
double zmax = xmax;
double eps_ = 1e-8;
double radius = 1.0;

/// Initial condition density
class IndataDensity : public Expression {
public:
  double &t;

  IndataDensity(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {
    double xx = x[0];
    double yy = x[1];

    double r = sqrt(pow(xx - 0.5, 2) + pow(yy - 1.75, 2));

    if ((yy <= 1) || (r <= 0.2)) {
      values[0] = rho1;
    } else {
      values[0] = rho2;
    }

  }
};

class IndataDensity3D : public Expression {
public:
  double &t;

  IndataDensity3D(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const {
    double xx = x[0];
    double yy = x[1];
    double zz = x[2];

    double r = sqrt(pow(xx - 0.5, 2) + pow(yy - 1.75, 2) + pow(zz - 0.5, 2));

    if ((yy <= 1) || (r <= 0.2)) {
      values[0] = rho1;
    } else {
      values[0] = rho2;
    }

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

class TopBottom : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && ((x[1] > ymax - eps_) || (x[1] < ymin + eps_));
  }
};

class LeftRight : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && ((x[0] > xmax - eps_) || (x[0] < xmin + eps_));
  }
};

class LeftRight_Z : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary && ((x[2] > zmax - eps_) || (x[2] < zmin + eps_));
  }
};

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



class CompleteBoundary : public SubDomain {
  bool inside(const Array<double> &x, bool on_boundary) const {
    return on_boundary;
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

void local_refinement(Mesh &mesh)
{
  std::cout << "local refinement" << std::endl;

  double y_cord = 1.8;
  double z_cord = 0.5;

  double tol_y = 0.1;
  double tol_z = 0.1;

  for (int k = 0; k < 1; ++k)
  {
    MeshFunction<bool> cell_markers(std::shared_ptr<Mesh>(&mesh, NoDeleter()),
                                    mesh.topology().dim(), false);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      // std::cout << "Refining cell " << std::endl;
      double x = c->midpoint().x();
      double y = c->midpoint().y();
      double z = c->midpoint().z();
      // double dist_0 = std::sqrt((x - x_cord0) * (x - x_cord0) + (y - y_cord0) * (y - y_cord0));
      // double dist_1_x = std::abs(x - x_cord1); //
      double dist_y = std::abs(y - y_cord); //
      double dist_z = std::abs(z - z_cord); //
      if ( ( dist_z < tol_z ) ){ 
        cell_markers[*c] = true;
        std::cout << "Refining cell " << std::endl;
      }
    }
    mesh = refine(mesh, cell_markers);
  }

  for (int k = 0; k < 1; ++k)
  {
    MeshFunction<bool> cell_markers(std::shared_ptr<Mesh>(&mesh, NoDeleter()),
                                    mesh.topology().dim(), false);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      // std::cout << "Refining cell " << std::endl;
      double x = c->midpoint().x();
      double y = c->midpoint().y();
      double z = c->midpoint().z();
      // double dist_0 = std::sqrt((x - x_cord0) * (x - x_cord0) + (y - y_cord0) * (y - y_cord0));
      // double dist_1_x = std::abs(x - x_cord1); //
      double dist_y = std::abs(y - y_cord); //
      double dist_z = std::abs(z - z_cord); //
      if ( ( dist_z < tol_z/2.0 ) ){ 
        cell_markers[*c] = true;
        std::cout << "Refining cell " << std::endl;
      }
    }
    mesh = refine(mesh, cell_markers);
  }

};