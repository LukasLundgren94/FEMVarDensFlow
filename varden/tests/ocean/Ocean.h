// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
// Copyright (c) 2025 Jonathan Wiskandt
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>

/* #include <dolfin.h> */
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/generation/BoxMesh.h>

using namespace dolfin;

double g = 9.81;
double eps_ = 1e-8;
double S_0 = 35.0; // 34.2 in firedrake
double T_0 = 0.0; // 1.0 in firedrake
double beta_S = 0.0008; // from default.yaml and from paper (8e-4)
double alpha_T = 0.4e-4;

// These are later computed in main.cpp using initialize_setup()
double xmin = -1.;
double xmax = -1.;
double ymin = -1.;
double ymax = -1.;
double zmin = ymin;
double zmax = ymax;
double x_cross_section = 21000.0; 
double x_r = 2000.0; // width of restoring region 
double skew_factor;
int   remove_rho_average;



struct ZetaCalculator
{


double zeta_active = 1.0;

  ZetaCalculator()
  {
      if (x_r <= 0.0) {
      zeta_active = 0.0;
    }
  }

  double calculate(double x_value) const
  {

    // from python code //
    // restoring_thickness = 2000.0 // means that x_r = 2000
    // xlayer = xmax - restoring_thickness
    // timescale = (60**2 * 24) / 10
    // # timescale = (60**2 * 24) / 24
    // from python code //
    double timescale = std::pow(60.0, 2.0) * 24.0 / 10.0;
    double frequency = 1. / timescale;

    double zeta0 = frequency;
    double x_ocean = xmax; // perhaps x_max / xmin or similar... (python code uses xlayer so this could be completely wrong)
    double zeta = zeta0 * std::max(0.0, (x_value - (x_ocean - x_r)) / x_r); // scaling layer is the max() part.

    return zeta_active*zeta; // if region is zero put zeta to zero...
  }
};

struct RestoringProfile_S
{

  RestoringProfile_S()
  {
  }

  double calculate(double y_value) const
  {
    // python code:
    // '34.5 + 1.0/2 * tanh(5 * pi * (-x[1] + 800.0) / ({ymax} - {ymin}))'
    // return 34.5 + 1.0 / 2. * std::tanh(5 * M_PI * (-y_value + 800.0) / (ymax - ymin));
    //     S_surf + (S_deep - S_surf) / (1 + np.exp((z - z_th)/delta))
    return 34.2 + (34.81 - 34.2) / (1. + std::exp((y_value - 770. + 1000.0)/47.)); 

  }
};

struct RestoringProfile_T
{

  RestoringProfile_T()
  {
  }

  double calculate(double y_value) const
  {
    // python code:
    // '(0.2 + (-1.6))/2 + (0.2 - (-1.6))/2 * tanh(5 * pi * (-x[1] + 800.0) / ({ymax} - {ymin}))'
    // return (0.2 + (-1.6)) / 2. + (0.2 - (-1.6)) / 2. * std::tanh(5. * M_PI * (-y_value + 800.0) / (ymax - ymin));
    //     T_surf + (T_deep - T_surf)  / (1 + np.exp((z      - z_th)/delta))
    return -0.676 + (0.345 - (-0.676)) / (1. + std::exp((y_value - 779. + 1000.0)/45.6));
  }
};

/// Initial condition temperature
class IndataTemperature : public Expression
{
public:
  double &t;
  RestoringProfile_T restoringProfile;

  IndataTemperature(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = restoringProfile.calculate(x[1]);
  }
};

class IndataPressure : public Expression
{
public:
  double &t;
  std::shared_ptr<double> T_bar_;
  std::shared_ptr<double> S_bar_;


  IndataPressure(double &t, std::shared_ptr<double> T_bar, std::shared_ptr<double> S_bar) : t(t), T_bar_(T_bar), S_bar_(S_bar)
  {

  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    
    double y = x[1];
    double y_atm = 0.0;
    // double S_bar = *S_bar_;
    // double T_bar = *T_bar_;
    // S_bar *= remove_rho_average;
    // T_bar *= remove_rho_average;

    // symbolic integration without f_u_source.
    // double offset = (y_atm*(-3.411918E-2)+((log(cosh((3.141592653589793*(x[1]-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*2.7468E-2)/3.141592653589793);
    // values[0] =  -offset + (x[1]*(-3.411918E-2)+((log(cosh((3.141592653589793*(x[1]-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*2.7468E-2)/3.141592653589793);



    // double offset = 0.133432312452523;
    // values[0] = -offset + y*2.158200000000009E-4+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*2.7468E-2)/3.141592653589793;;

    
    // double offset = 1.507240123765541;
    // values[0] = -offset +  y*3.649319999999972E-3+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793;

    // with new method:
    y = y_atm;
    double offset = y_atm*3.649319999999972E-3+((log(cosh((3.141592653589793*(y_atm-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793-y_atm*(tanh((3.141592653589793*(y_atm-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*(skew_factor-1.0)*3.924000000000001E-4+y_atm*(tanh((3.141592653589793*(y_atm-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*(skew_factor-1.0)*7.848000000000001E-3;

    y = x[1];
    values[0] = -offset + y*3.649319999999972E-3+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793-y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*(skew_factor-1.0)*3.924000000000001E-4+y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*(skew_factor-1.0)*7.848000000000001E-3;

    // new method, with no f_u_source:
    // y = y_atm;
    // double offset = y*(-2.7103068E-1)+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793+y*(skew_factor-1.0)*(S_bar+tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*7.848000000000001E-3-y*(skew_factor-1.0)*(T_bar+tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*3.924000000000001E-4;

    // y = x[1];
    // values[0] = -offset + y*(-2.7103068E-1)+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793+y*(skew_factor-1.0)*(S_bar+tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*7.848000000000001E-3-y*(skew_factor-1.0)*(T_bar+tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*3.924000000000001E-4;
    
    // pressure shift is handled by regular boundary term.
    y = y_atm;
    offset = y*3.649319999999972E-3+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793-y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*(skew_factor-1.0)*8.71303029725823E-20+y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*(skew_factor-1.0)*1.742606059451646E-18;

    y = x[1];
    values[0] = -offset + y*3.649319999999972E-3+((log(cosh((3.141592653589793*(y-8.0E+2))/2.0E+2))-log(cosh(3.141592653589793*4.0)))*7.141680000000001E-1)/3.141592653589793-y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)*(9.0/1.0E+1)+7.0/1.0E+1)*(skew_factor-1.0)*8.71303029725823E-20+y*(tanh((3.141592653589793*(y-8.0E+2))/2.0E+2)/2.0-6.9E+1/2.0)*(skew_factor-1.0)*1.742606059451646E-18;

   

    // std::cout << "T_bar = " << T_bar << std::endl;
    // std::cout << "S_bar = " << S_bar << std::endl;
  }
};

/// Initial condition salt
class IndataSalt : public Expression
{
public:
  double &t;
  RestoringProfile_S restoringProfile;

  IndataSalt(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = restoringProfile.calculate(x[1]);
  }
};

/// Initial condition density
class IndataDensity : public Expression
{
public:
  double &t;

  IndataDensity(double &t) : t(t)
  {
  }

  void eval(Array<double> &values, const Array<double> &x) const
  {

    // constant density benchmark
    values[0] = 1.0;
  }
};

class Forcing_S : public Expression
{
public:
  double &t;
  ZetaCalculator zetaCalc;
  RestoringProfile_S restoringProfile;

  Forcing_S(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const
  {
    double S_restoring = restoringProfile.calculate(x[1]);
    double zeta = zetaCalc.calculate(x[0]);
    values[0] = -zeta * S_restoring;
  }
};

class Reaction_S : public Expression
{
public:
  double &t;
  ZetaCalculator zetaCalc;
  RestoringProfile_S restoringProfile;

  Reaction_S(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const
  {
    // double S_restoring = restoringProfile.calculate(x[1]);
    double zeta = zetaCalc.calculate(x[0]);
    values[0] = zeta;
  }
};

class Forcing_T : public Expression
{
public:
  double &t;
  RestoringProfile_T restoringProfile;
  ZetaCalculator zetaCalc;

  Forcing_T(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const
  {
    double T_restoring = restoringProfile.calculate(x[1]);
    double zeta = zetaCalc.calculate(x[0]);
    values[0] = -zeta * T_restoring;
  }
};

class Reaction_T : public Expression
{
public:
  double &t;
  RestoringProfile_T restoringProfile;
  ZetaCalculator zetaCalc;

  Reaction_T(double &t) : t(t) {}

  void eval(Array<double> &values, const Array<double> &x) const
  {
    // double T_restoring = restoringProfile.calculate(x[1]);
    double zeta = zetaCalc.calculate(x[0]);
    values[0] = zeta;
  }
};

// Define noslip domain
class NoslipDomain : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && (x[1] < ymin + eps_ || x[1] > ymax - eps_);
  }
};

// Define noslip domain
class SlipDomain : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && (x[0] < xmin + eps_ || x[0] > xmax - eps_);
  }
};

class TopBottom : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && ((x[1] > ymax - eps_) || (x[1] < ymin + eps_));
  }
};

class LeftRight : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && ((x[0] > xmax - eps_) || (x[0] < xmin + eps_));
  }
};

class Left : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && (x[0] < xmin + eps_);
  }
};

class Right : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && ((x[0] > xmax - eps_));
  }
};

class Top : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && (x[1] > ymax - eps_);
  }
};

class Bottom : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && ((x[1] < ymin + eps_));
  }
};

// Define top domain
class TopWall : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && x[1] > ymax - eps_;
  }
};

// Define top domain
class ZWall : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && x[2] > zmax - eps_  || x[2] < zmin + eps_;
  }
};

// Define bottom domain
class BottomWall : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && x[1] < ymin + eps_;
  }
};

class CornersOutflowTop : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && ( ( (x[1] > (ymax - eps_) ) && (x[0] > (xmax - 500.) ) )   );
  }
};

class CornersOutflowBottom : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && (  (x[1] < ymin + eps_ && x[0] > xmax - 500.) );
  }
};


class CrossSection : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return std::abs(x[0]-x_cross_section) < eps_;
  }
};


// Sub domain for Periodic boundary condition for x and z walls
class PeriodicBoundary : public SubDomain
{

  // double xmin = 0.0;
  // double xmax = 10.0;
  // double ymin = 0.0;
  // double ymax = 2.0;
  // double zmin = 0.0;
  // double zmax = 3.0;
  // double eps = DOLFIN_EPS_LARGE;
  double eps = eps_;

  // Left boundary is "target domain" G
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return (
               (x[0] < xmin + eps)) and
           on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double> &x, Array<double> &y) const
  {
    y[0] = x[0] - (xmax - xmin);
    y[1] = x[1];
  }
};

/// Local refinement
void local_refinement(Mesh &mesh)
{
  std::cout << "local refinement" << std::endl;

  double x_cord0 = 20000.0;
  double y_cord0 = 950.0;

  double x_cord1 = 32000.0;
  double y_cord1 = 750.0;

  // current refinement from 550 to 950
  // lets switch to 600 to 900, 750 +- 150

  double tol_bottom = 50.;
  double tol1 = 50.;
  double tol2_x = 50.;
  double tol2_y = 150.;

  for (int k = 0; k < 1; ++k)
  {
    MeshFunction<bool> cell_markers(std::shared_ptr<Mesh>(&mesh, NoDeleter()),
                                    mesh.topology().dim(), false);
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      // std::cout << "Refining cell " << std::endl;
      double x = c->midpoint().x();
      double y = c->midpoint().y();
      double dist_0 = std::sqrt((x - x_cord0) * (x - x_cord0) + (y - y_cord0) * (y - y_cord0));
      double dist_1_x = std::abs(x - x_cord1); //
      double dist_1_y = std::abs(y - y_cord1); //
      if ( (dist_0 < tol1 && x>=x_cord0) || ( dist_1_y < tol2_y ) ){ // 
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
      double dist_0 = std::sqrt((x - x_cord0) * (x - x_cord0) + (y - y_cord0) * (y - y_cord0));
      double dist_1_x = std::abs(x - x_cord1); //
      double dist_1_y = std::abs(y - y_cord1); // 
      double dist_bottom = std::abs(y);
      if ( (dist_0 < tol1 && x>=x_cord0) || (dist_bottom < tol_bottom) ){ // 
        cell_markers[*c] = true;
        std::cout << "Refining cell " << std::endl;
      }
    }
    // mesh = refine(mesh, cell_markers);
  }
};

class CompleteBoundary : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary;
  }
};

class SlipBC_marker : public Expression
{
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = 1.0;

    /*    if (((x[0] > xmin  && x[0] < xmax -0.1  - eps_) &&
            (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmin  && x[0] < xmax  - eps_ - 0.1) ) {
          values[0] = 1.0;
        }*/
  }
};

class SlipBC_marker_fn : public Expression
{
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = 1.0;

    /*    if (((x[0] > xmax - 0.1  && x[0] < xmax  - eps_) &&
            (x[1] > ymin + eps_ && x[1] < ymax - eps_) ) || (x[0] > xmax - 0.1  && x[0] < xmax  - eps_ ) ) {
          values[0] = 1.0;
        }*/
  }
};

// Implement complicated 3-equation model later
class NeumannBC_Temperature : public Expression
{

  void eval(Array<double> &values, const Array<double> &x) const
  {

    values[0] = 1.0;
  }
};

void mark_facet_direction(std::shared_ptr<MeshFunction<std::size_t>> mesh_function, int direction, std::size_t value,
                          double tol = 1.e-12)
{

  // MeshFunction should be defined as a function on all facets so maybe everything is fine.
  // std::size_t n_facets = mesh.num_facets();

  // std::vector<std::size_t> Fmarker_arr(mesh_function->num_entities(d-1)); // dof_V_.size()

  // FacetIterator in c++:

  //   for (FacetIterator cell(*mesh_); !cell.end(); ++cell)
  // {
  //   for (std::size_t node = 0; node < num_cell_nodes; ++node, ++ii)
  //   {
  //     double ui, ni, n_scal = 0.0;
  //     //facet normal
  //     normal = Fn[ii] = ???;
  //   }
  //   offset_u += Fu_->function_space()->dofmap()->max_element_dofs();
  // }

  // facets are defined like this...
  // MeshValueCollection can be used to create a new MeshFunction.
  // MeshValueCollection<std::size_t> facets(mesh, dim-1);
  std::shared_ptr<const Mesh> mesh = mesh_function->mesh();
  std::size_t dim = mesh_function->dim();

  // MeshValueCollection<std::size_t> valueCollection(mesh, dim);
  MeshValueCollection<std::size_t> valueCollection(*mesh_function);

  for (CellIterator cell(*mesh_function->mesh()); !cell.end(); ++cell)
  {
    // iterate over all facets
    std::size_t ii = 0;
    for (FacetIterator ff(*cell); !ff.end(); ++ff, ++ii)
    {
      if (ff->exterior())
      {
        auto normal = ff->normal();
        if ((direction == 1) && normal.y() > tol)
        {
          valueCollection.set_value(cell->index(), ii, value);
        }
        else if ((direction == -1) && normal.y() < -tol)
        {
          valueCollection.set_value(cell->index(), ii, value);
        }
        else if ((direction == 0) && std::abs(normal.y()) < tol)
        {
          valueCollection.set_value(cell->index(), ii, value);
        }
      }
    }
  }

  *mesh_function = valueCollection;
}

void fix_max_min(std::shared_ptr<Mesh> mesh)
{

  std::size_t dim = mesh->topology().dim();

  // extract max and min of mesh
  std::vector<double> coordinates = mesh->coordinates();

  auto x_points = std::vector<double>(coordinates.size() / dim);
  auto y_points = std::vector<double>(coordinates.size() / dim);
  auto z_points = std::vector<double>(coordinates.size() / dim);

  // put all even elements of coordinates into x_points and put all odd elements of coordinates into y_points:
  for (std::size_t i = 0; i < coordinates.size(); ++i)
  {
    if (i % dim == 0)
    {
      x_points[i / dim] = coordinates[i];
    }
    else if(i % dim == 1)
    {
      y_points[(i - 1) / dim] = coordinates[i];
    }
    else if(i % dim == 2)
    {
      z_points[(i - 2) / dim] = coordinates[i];
    }
  }

  // Initialize local min and max
  double localMin_x = std::numeric_limits<double>::max();
  double localMax_x = std::numeric_limits<double>::lowest();
  double localMin_y = std::numeric_limits<double>::max();
  double localMax_y = std::numeric_limits<double>::lowest();
  double localMin_z = std::numeric_limits<double>::max();
  double localMax_z = std::numeric_limits<double>::lowest();

  if (!x_points.empty())
  {
    localMin_x = *std::min_element(x_points.begin(), x_points.end());
    localMax_x = *std::max_element(x_points.begin(), x_points.end());
  }

  if (!y_points.empty())
  {
    localMin_y = *std::min_element(y_points.begin(), y_points.end());
    localMax_y = *std::max_element(y_points.begin(), y_points.end());
  }
  if (!z_points.empty())
  {
    localMin_z = *std::min_element(z_points.begin(), z_points.end());
    localMax_z = *std::max_element(z_points.begin(), z_points.end());
  }
  // std::cout << "Local Max: " << localMax << std::endl;

  double globalMin_x, globalMax_x, globalMin_y, globalMax_y, globalMin_z, globalMax_z;
  // Reduce to find global min and max
  globalMin_x = dolfin::MPI::min(mesh->mpi_comm(), localMin_x);
  globalMax_x = dolfin::MPI::max(mesh->mpi_comm(), localMax_x);
  globalMin_y = dolfin::MPI::min(mesh->mpi_comm(), localMin_y);
  globalMax_y = dolfin::MPI::max(mesh->mpi_comm(), localMax_y);
  globalMin_z = dolfin::MPI::min(mesh->mpi_comm(), localMin_z);
  globalMax_z = dolfin::MPI::max(mesh->mpi_comm(), localMax_z);

  // print global min and max of x and y:
  // std::cout << "Global Min_x: " << globalMin_x << std::endl;
  // std::cout << "Global Max_x: " << globalMax_x << std::endl;
  // std::cout << "Global Min_y: " << globalMin_y << std::endl;
  // std::cout << "Global Max_y: " << globalMax_y << std::endl;

  xmin = globalMin_x;
  xmax = globalMax_x;
  ymin = globalMin_y;
  ymax = globalMax_y;
  zmin = globalMin_z;
  zmax = globalMax_z;
}

// Python code:
// def mark_facet_direction(mesh_function, direction, value,
//  midpoint_condition=None,
//                     //  tol=1e-12):
//                      # check if we are dealing with facets
// assert mesh_function.dim() == mesh_function.mesh().topology().dim()-1, \
    //     "Expected arg: {mesh_function} to contain facets"
// allowed_directions = {"up": 1, "down": -1, "sides": 0}
// if direction in allowed_directions:
//     direction = allowed_directions[direction]
// elif direction in allowed_directions.values():
//     direction = direction
// else:
//     raise RuntimeError("Unknown direction `{direction}`")
// ind = mesh_function.mesh().geometric_dimension() - 1
// for facet in facets(mesh_function.mesh()):
//     if not facet.exterior():
//         continue
//     if midpoint_condition and not midpoint_condition(facet.midpoint()):
//         continue
//     n = facet.normal()
//     if direction == 1 and n[ind] > tol:
//         mesh_function[facet] = value
//     elif direction == -1 and n[ind] < -tol:
//         mesh_function[facet] = value
//     elif direction == 0 and abs(n[ind]) < tol:
//         mesh_function[facet] = value
