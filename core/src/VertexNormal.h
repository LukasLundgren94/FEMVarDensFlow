// Copyright (C) 2007-2017 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008-2010. 
// Modifed by Christian Helanow, 2023.
// Modified by Lukas Lundgren 2025. 
//  
// First added:  2007-05-01
// Last changed: 2017-01-01

#ifndef __VERTEXNORMAL_H
#define __VERTEXNORMAL_H

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/function/Function.h>
#include <algorithm>

///
namespace dolfin
{
  class DofMap;
  class Function;
  class FunctionSpace;
  class Mesh;
  class SubDomain;
  class Form;
  class GenericMatrix;
  class GenericVector;
  class DirichletBC;
  class Matrix;
  class Vector;
  class BoundaryNormal;
  class BoundaryMesh;
  template<typename T> class MeshFunction;
}

///
using dolfin::DofMap;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::Mesh;
using dolfin::BoundaryMesh;
using dolfin::SubDomain;
using dolfin::Form;
using dolfin::GenericMatrix;
using dolfin::GenericVector;
using dolfin::CellIterator;
using dolfin::VertexIterator;
using dolfin::Vertex;
using dolfin::Facet;
using dolfin::Cell;
using dolfin::MeshFunction;
using dolfin::Variable;
using dolfin::Parameters;
using dolfin::Matrix;
using dolfin::Vector;
using dolfin::DirichletBC;
using dolfin::File;
using dolfin::MPI;
using dolfin::set_log_active;


///
namespace fem_varden_flow
{
  class DofMappings;
  class VertexNormal;

  class VertexNormal : public Variable
{
public:
  // Create normal, tangents for the boundary of mesh
  VertexNormal(std::shared_ptr<const Mesh> mesh);

  ///
  ~VertexNormal();

  ///
  void clear();

  // Define node type: 1 surface, 2 edge, 3 surface
  MeshFunction<std::size_t> Fnode_type_;

  // Default parameter values
  static Parameters default_parameters()
  {
    Parameters p("node_normal");
    p.add("alpha_max", DOLFIN_PI / 4.0);
    return p;
  }

  // Define vectors for the normal and tangents
  MeshFunction<double> * Fnormal;
  MeshFunction<double> * Ftau1;
  MeshFunction<double> * Ftau2;

private:

  // Compute normals to the boundary nodes
  void compute_normal();

  // Compute tangents according to the normal
  void compute_tangents();

  ///
  void compute_tangent(std::vector<double> const & normal,
                       std::vector<double> & tau1, std::vector<double> & tau2);
  ///
  void get_facet_normal(Vertex const& v, std::vector<double>& n_weighted,
                        double& sum_weights);

  ///
  std::shared_ptr<const Mesh> mesh_;

  ///
  MeshFunction<std::size_t> cell_map_;
  MeshFunction<std::size_t> vertex_map_;

  ///
  std::shared_ptr<BoundaryMesh> boundary_;

  //
  class SharedData
  {
  public:

    // Constructor
    SharedData();

    std::vector<std::pair<std::size_t, std::size_t>> processes;
    std::vector<double> weight;
    std::vector<std::size_t> glob_id;
    std::vector<std::vector<double>> n_block;
  };

};

}

#endif
