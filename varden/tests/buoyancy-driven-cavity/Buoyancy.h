// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>

/* #include <dolfin.h> */
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/generation/BoxMesh.h>


using namespace dolfin;

double d = 1.0;
double b = 0.1;
double rho1 = 3.0;
double rho2 = 1.0;
double t_warm = 1.0;
double t_cold = 0.0;
double g = 1.0;
double xmin = -d/2.;
double xmax = d/2.;
double ymin = -d/2.;
double ymax = d/2.;
double zmin = ymin;
double zmax = ymax;
double eps_ = 1e-8;
double radius = 1.0;


/// Initial condition temperature
class IndataTemperature : public Expression
{
public:
  double& t;

 IndataTemperature(double& t):t(t){
 }
  void eval(Array<double>& values, const Array<double>& x) const
  {


    values[0] =  0.;

  }
};

/// Initial condition density
class IndataDensity : public Expression
{
 public:
  double& t;

 IndataDensity(double& t):t(t){

  }
  
  void eval(Array<double>& values, const Array<double>& x) const
  {

    // constant density benchmark
    values[0] =  1.0;

  }
};

// Define noslip domain
class NoslipDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return 
      on_boundary && (x[1] < ymin + eps_ || x[1] > ymax - eps_);
  }
};

// Define noslip domain
class SlipDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return 
      on_boundary && (x[0] < xmin + eps_ || x[0] > xmax - eps_);
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

class Left : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary &&  (x[0] < xmin + eps_) ;
  }
};

class Right : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && ( (x[0] > xmax - eps_)  );
  }
};


class Top : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && ( (x[1] > ymax - eps_)  );
  }
};

class Bottom : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && (  (x[1] < ymin + eps_) );
  }
};


// Define top domain
class TopWall : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && x[1] > ymax - eps_;
  }
};

// Define bottom domain
class BottomWall : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
    return  
      on_boundary && x[1] < ymin + eps_;
  }
};


// Sub domain for Periodic boundary condition for x and z walls
class PeriodicBoundary : public SubDomain
{

  //double xmin = 0.0;
  //double xmax = 10.0;
  //double ymin = 0.0;
  //double ymax = 2.0;
  //double zmin = 0.0;
  //double zmax = 3.0;
  //double eps = DOLFIN_EPS_LARGE;
  double eps = eps_;


  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return
      (
       (x[0] < xmin + eps)
       )
      and
      on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
        y[0] = x[0] - (xmax - xmin);
        y[1] = x[1];
  }
};

/// Local refinement
void local_refinement(Mesh & mesh)
{
  for (int k = 1 ; k < 1; ++k)
    {
      MeshFunction<bool> cell_markers(std::shared_ptr<Mesh>(&mesh, NoDeleter()),
				      mesh.topology().dim(), false);
      for (CellIterator c(mesh); !c.end(); ++c)
	{
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
        (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmin  && x[0] < xmax  - eps_ - 0.1) ) {
      values[0] = 1.0;
    }*/
  }
};

class SlipBC_marker_fn : public Expression {
  void eval(Array<double> &values, const Array<double> &x) const {
    values[0] = 0.0;

/*    if (((x[0] > xmax - 0.1  && x[0] < xmax  - eps_) &&
        (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmax - 0.1  && x[0] < xmax  - eps_ ) ) {
      values[0] = 1.0;
    }*/
  }
};

