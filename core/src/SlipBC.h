// Copyright (C) 2007-15 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Existing code for Dirichlet BC is used
//
// Modified by Niclas Jansson, 2008-2009.
// Modified by Lukas Lundgren 2021-2025.
// Modified by Christian Helanow, 2023.
//
// First added:  2007-05-01
// Last changed: 2015-10-10

#ifndef __SLIPBC_H
#define __SLIPBC_H

#include <boost/shared_ptr.hpp>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/log/Logger.h>
#include <dolfin/log/Progress.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/mesh/MeshEntityIterator.h>

///
namespace dolfin
{
  class FunctionSpace;
  class Function;
  class Constant;
  class Timer;
  class Progress;
  class Logger;
  class Mesh;
  class BoundaryMesh;
  class SubDomain;
  class DofMap;
  class Form;
  class File;
  class SparsityPatternBuilder;
  class DirichletBC;
  class Matrix;
  class Vector;
  template<typename T> class MeshFunction;
}

///
using dolfin::FiniteElement;
using dolfin::DofMap;
using dolfin::GenericDofMap;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::Mesh;
using dolfin::SubDomain;
using dolfin::Form;
using dolfin::GenericMatrix;
using dolfin::GenericVector;
/* using dolfin::FacetIterator; */
using dolfin::CellIterator;
using dolfin::MeshFunction;
using dolfin::Hierarchical;
using dolfin::Variable;
using dolfin::Parameters;
using dolfin::Matrix;
using dolfin::Vector;
using dolfin::DirichletBC;
using dolfin::File;
using dolfin::Timer;
using dolfin::Logger;
using dolfin::Progress;
using dolfin::warning;

///
namespace fem_varden_flow
{
  class DofMappings;
  class VertexNormal;
  class NodeNormal;
  
  class SlipBC: public Hierarchical<SlipBC>, public Variable  
  {
  public:
    typedef std::unordered_map<std::size_t, double> Map;

    /// Create boundary condition for sub domain
    SlipBC(std::shared_ptr<const FunctionSpace> V,
	   std::shared_ptr<const SubDomain> sub_domain);

    SlipBC(std::shared_ptr<const FunctionSpace> V,
	   std::shared_ptr<const FunctionSpace> V_CG1,
	   std::shared_ptr<const SubDomain> sub_domain);

    SlipBC(std::shared_ptr<const FunctionSpace> V,
               std::shared_ptr<const FunctionSpace> V_CG1,
               std::shared_ptr<const MeshFunction<std::size_t>> sub_domains,
               std::size_t sub_domain);

    /// Destructor
    ~SlipBC();

    void init();

    void apply(GenericMatrix& A, GenericVector& b);
    
    void apply_system(GenericMatrix& A, GenericMatrix& A01, GenericVector& b);

    void apply_inflow_system(GenericMatrix& A, GenericMatrix& A01, GenericVector& b, GenericVector& g);

    void apply_inflow(GenericMatrix& A, GenericVector& b, GenericVector& g);

    void zero(GenericMatrix& A);

    void gather(std::vector<std::vector<std::size_t>>& v_slip_dofs) const;

    /// Return boundary markers (facets stored as pairs of cells and local facet numbers)
    const std::vector<std::pair<std::size_t, std::size_t> >& markers();

    // Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("slip_bc");
      p.add("skip_corner", false);
      p.add("dual", false);
      p.add("psi1", 0.);
      p.add("psi2", 0.);
      p.add("psi3", 0.);
      return p;
    }

    inline void save_normals()
    {
      File fnormal("normal.pvd", "compressed");
      fnormal << *Fnormal_;
      fnormal << *Ftau1_;
      fnormal << *Ftau2_;
    }
  private:
    class LocalData;

    ///
    std::shared_ptr<const FunctionSpace> V_;
    std::shared_ptr<const FunctionSpace> V_CG1_;

    void applySlipBC(Matrix* A, Matrix* As, Vector* b, Vector* bs,
		     const Mesh& mesh, const NodeNormal& normal_node,
		     std::size_t node, std::vector<std::size_t>& dofs) const;

    void applySlipBC(GenericMatrix& A, GenericVector& b,
		     std::vector<std::size_t>& dofs);

    void applySystemSlipBC(GenericMatrix& A, GenericMatrix& A01, GenericVector& b,
         std::vector<std::size_t>& dofs);

    void applySystemInflowBC(GenericMatrix& A, GenericMatrix& A01, GenericVector& b,
         std::vector<std::size_t>& dofs, GenericVector& g);

    void applyZeroSlipBC(GenericMatrix& A,
         std::vector<std::size_t>& dofs);
    
    void applyInflowBC(GenericMatrix& A, GenericVector& b,
		     std::vector<std::size_t>& dofs, GenericVector& g);

    // Initialize normal vectors
    void init_normals();

    // Initialize sub domain markers from sub domain
    void init_from_sub_domain(std::shared_ptr<const SubDomain> sub_domain) const;

    void init_from_mesh_function(const MeshFunction<std::size_t>& sub_domains,
				 std::size_t sub_domain) const;

    // Do: A(row,col) = value   using setblock not setvalue
    inline void Aset(GenericMatrix& A, dolfin::la_index row, dolfin::la_index col,
		     double value) const
    {
      A.set(&value, 1, &row, 1, &col);
    }
    ;

    // Do: b(row) = value   using setblock not setvalue
    inline void bset(GenericVector& b, dolfin::la_index const& row,
		     double const& value) const
    {
      b.set_local(&value, 1, &row);
    }
    ;

    inline void bget(GenericVector const& b, dolfin::la_index const& row,
		     double& value) const
    {
      b.get_local(&value, 1, &row);
    }
    ;

    // Node normal and tangents
    std::shared_ptr<Function> Fnormal_;
    std::shared_ptr<Function> Ftau1_;
    std::shared_ptr<Function> Ftau2_;

    // User defined sub domain
    std::shared_ptr<const SubDomain> user_sub_domain_;

    // Sub domain markers (if any)
    std::shared_ptr<MeshFunction<std::size_t>> sub_domains_;
    std::shared_ptr<const MeshFunction<std::size_t>> user_mesh_function_;
    std::size_t user_sub_domain_marker_;

    // Boundary facets, stored by facet index (local to process)
    mutable std::vector<std::size_t> facets_;

    std::shared_ptr<Matrix> As_;
    std::shared_ptr<Vector> bs_;

    std::shared_ptr<Matrix> As_zero_;

    std::shared_ptr<DirichletBC> dbc;
    std::shared_ptr<const Mesh> mesh_;

    std::vector<std::vector<std::size_t>> v_slip_dofs_;

    // Compute boundary values for facet (topological approach)
    void compute_bc_topological();

    void init_facets(const MPI_Comm mpi_comm) const;

    // Local data for application of boundary conditions
    class LocalData
    {
    public:

      // Constructor
      LocalData(const FunctionSpace& V);

      // Coefficients
      std::vector<double> w;

      // Facet dofs
      std::vector<std::size_t> facet_dofs;

      // Coordinates for dofs
      boost::multi_array<double, 2> coordinates;

    };
  };
}

#endif
