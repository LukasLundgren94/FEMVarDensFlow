// Copyright (C) 2007-2017 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008-2009. 
// Modified by Lukas Lundgren 2021-2025.
// Modified by Christian Helanow, 2023.
//
// First added:  2007-05-01
// Last changed: 2017-01-25

#ifndef __NODENORMAL_H
#define __NODENORMAL_H

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include "VertexNormal.h"

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
using dolfin::MeshFunction;
using dolfin::Hierarchical;
using dolfin::Variable;
using dolfin::Parameters;
using dolfin::Matrix;
using dolfin::Vector;
using dolfin::DirichletBC;
using dolfin::File;
using dolfin::Array;
using dolfin::Constant;
using dolfin::Cell;




///
namespace fem_varden_flow
{
  class DofMappings;
  class VertexNormal;
  
  class NodeNormal: public Variable
  {
  public:
    // Create normal, tangents for the boundary of mesh
    ///
    NodeNormal(std::shared_ptr<const FunctionSpace> V,
	       std::shared_ptr<const FunctionSpace> V_CG1);

    ///
    NodeNormal(std::shared_ptr<const FunctionSpace> V_CG1);

    ///
    ~NodeNormal();

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

    ///
    inline Function& normal()
    {
      return *Fnormal_;
    }

    ///
    inline Function& tau1()
    {
      return *Ftau1_;
    }

    ///
    inline Function& tau2()
    {
      return *Ftau2_;
    }

  private:

    // Define vectors for the normal and tangents
    std::shared_ptr<Function> Fnormal_;
    std::shared_ptr<Function> Ftau1_;
    std::shared_ptr<Function> Ftau2_;

    void normalize();
    void normalize_vector(std::vector<double>& v,
                          std::vector<double>& magnitudes);

    // Compute normals to the boundary nodes
    void compute_CG1(Function& Fnormal, Function& Ftau1, Function& Ftau2);
    void compute();
    void init();
    

    std::shared_ptr<const FunctionSpace> V_;
    std::shared_ptr<const FunctionSpace> V_CG1_;
    std::shared_ptr<const Mesh> mesh_;

    MeshFunction<std::size_t> cell_map_;
    MeshFunction<std::size_t> vertex_map_;

    std::shared_ptr<BoundaryMesh> boundary_;
    std::shared_ptr<VertexNormal> vertex_normal_;

  };

}

#endif
