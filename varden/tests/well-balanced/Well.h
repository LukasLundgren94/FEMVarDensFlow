// Copyright (c) 2017 Murtazo Nazarov
// Copyright (c) 2023-2025 Lukas Lundgren
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
#include <ConfigurableParameters.h>
#include <memory>

using namespace dolfin;

double eps_ = 1e-8;
double shift = 10.; // = 1

// These are later computed using in main.cpp using initialize_setup()
double xmin = -1.;
double xmax = 1.;
double ymin = -1.;
double ymax = 1.;
double zmin = ymin;
double zmax = ymax;
double skew_rho = 0.0; // is set in main.cpp




/// Initial condition temperature
class IndataTemperature : public Expression
{
public:
  double &t;

  IndataTemperature(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    // values[0] = (x[1]+1)*0.5;
    // values[0] = shift -( 0.5*tanh(5*x[1])+shift); // ??
    values[0] =( 0.5*tanh(5*x[1])) + shift ; 
  }
};

/// Initial condition temperature
class IndataTemperatureLinear : public Expression
{
public:
  double &t;

  IndataTemperatureLinear(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = (x[1]+1)*0.5;
    // values[0] = 0.5*tanh(5*x[1])+1;
  }
};


class IndataPressure : public Expression
{
public:
  double &t;
  std::shared_ptr<double> T_bar_;

  IndataPressure(double &t, std::shared_ptr<double> T_bar) : t(t), T_bar_(T_bar)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    double T_bar = *T_bar_;
    T_bar = 0.0;
    // stretch domain so that I know that I understand everything.
    //  values[0] = -offset + (x[1] + 1)*(x[1] + 1)*0.25;
    
    // double T_bar = T; // for y_max = 1
    // double T_bar = 1.016666515343873e+01; // for y_max = 2

    // why shift -4!??!? 4 = area of domain???????????
    double offset = 0.1889094982841017; // was this previously without skew_rho method
    // double offset = 6.801933650918945e-02;//6.801933650918905e-02; // shift = 10, and with skew_rho method, y_max = 1
    //  double offset = 5.188908590246859e+00; // shift = 10, and with skew_rho method, y_max = 2
    //  double offset = 6.801933650918905e-02;

    // (1-C_skew_rho_factor)*div(v)*( (T_ - T_bar)*(f_T[0]*x[0] + f_T[1]*x[1])
                      
    double f_T_1 = 1.0;
    double T = 0.5*tanh(5*x[1]) + shift - 0*T_bar; 
    double force_shift = (1-skew_rho)*x[1]*f_T_1*T;

    values[0] = (-offset + (1.0/10.0)*log(cosh(5*x[1])) + shift*x[1] - eps_*force_shift);
    
    // std::cout << "T_bar = " << T_bar << std::endl;
  }
};

class IndataPressureLinear : public Expression
{
public:
  double &t;

  IndataPressureLinear(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    // offset is chosen so that pressure is zero at the top
    double offset = 1./3.;
     values[0] = -offset + (x[1] + 1)*(x[1] + 1)*0.25;
  }
};

/// Initial condition salt
class IndataSalt : public Expression
{
public:
  double &t;

  IndataSalt(double &t) : t(t)
  {
  }
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = (x[1]+1)*0.5;
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

// Define bottom domain
class BottomWall : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return on_boundary && x[1] < ymin + eps_;
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
  for (int k = 1; k < 1; ++k)
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
    values[0] = 0.0;

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
    values[0] = 0.0;

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

void initialize_setup(std::shared_ptr<Mesh> mesh)
{

  std::size_t dim = mesh->topology().dim();

  // extract max and min of mesh
  std::vector<double> coordinates = mesh->coordinates();

  auto x_points = std::vector<double>(coordinates.size() / dim);
  auto y_points = std::vector<double>(coordinates.size() / dim);

  // put all even elements of coordinates into x_points and put all odd elements of coordinates into y_points:
  for (std::size_t i = 0; i < coordinates.size(); ++i)
  {
    if (i % 2 == 0)
    {
      x_points[i / 2] = coordinates[i];
    }
    else
    {
      y_points[(i - 1) / 2] = coordinates[i];
    }
  }

  // Initialize local min and max
  double localMin_x = std::numeric_limits<double>::max();
  double localMax_x = std::numeric_limits<double>::lowest();
  double localMin_y = std::numeric_limits<double>::max();
  double localMax_y = std::numeric_limits<double>::lowest();

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
  // std::cout << "Local Max: " << localMax << std::endl;

  double globalMin_x, globalMax_x, globalMin_y, globalMax_y;
  // Reduce to find global min and max
  globalMin_x = dolfin::MPI::min(mesh->mpi_comm(), localMin_x);
  globalMax_x = dolfin::MPI::max(mesh->mpi_comm(), localMax_x);
  globalMin_y = dolfin::MPI::min(mesh->mpi_comm(), localMin_y);
  globalMax_y = dolfin::MPI::max(mesh->mpi_comm(), localMax_y);

  // print global min and max of x and y:
  std::cout << "Global Min_x: " << globalMin_x << std::endl;
  std::cout << "Global Max_x: " << globalMax_x << std::endl;
  std::cout << "Global Min_y: " << globalMin_y << std::endl;
  std::cout << "Global Max_y: " << globalMax_y << std::endl;

  xmin = globalMin_x;
  xmax = globalMax_x;
  ymin = globalMin_y;
  ymax = globalMax_y;
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
