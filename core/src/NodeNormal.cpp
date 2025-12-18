// Copyright (C) 2007-2017 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008-2010.
// Modified by Lukas Lundgren 2021-2025.
// Modified by Christian Helanow, 2023.
//
// First added:  2007-05-01
// Last changed: 2017-01-25

//#include <dolfin.h>
#include <algorithm>
#include <unordered_map>

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/Constant.h>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/math/basic.h>
#include <dolfin/log/log.h>
#include <dolfin/common/constants.h>

#include "NodeNormal.h"

using namespace fem_varden_flow;


//-----------------------------------------------------------------------------
NodeNormal::NodeNormal(std::shared_ptr<const FunctionSpace> V_CG1) :
    V_(V_CG1),
    V_CG1_(V_CG1),
    mesh_(V_CG1->mesh())
{
  init();
}

//-----------------------------------------------------------------------------
NodeNormal::NodeNormal(std::shared_ptr<const FunctionSpace> V,
                       std::shared_ptr<const FunctionSpace> V_CG1) :
    V_(V),
    V_CG1_(V_CG1),
    mesh_(V_CG1->mesh())
{
  init();
}

//-----------------------------------------------------------------------------
void NodeNormal::init()
{
  Fnormal_ = std::make_shared<Function>(V_);
  Ftau1_ = std::make_shared<Function>(V_);
  Ftau2_ = std::make_shared<Function>(V_);

  boundary_ = std::make_shared<BoundaryMesh>(*mesh_, "exterior");
  vertex_map_ = boundary_->entity_map(0);
  cell_map_ = boundary_->entity_map(boundary_->topology().dim());

  // CH Note:
	// since UnitVertexNormal is *not* normalized by default (specified with
	// argument), and UnitNodeNormal::normalize() is called *after*
	// interpolation between the vertex normals, there will be a slight
	// difference between the non-vertex normals, which depend on the
	// interpolation between values, i.e., interpolation between unit
	// normals at the vertices or non-unit normals at the vertices.
	// not necessary if normalized globally by.
	// Here we use the normalization of UnitVertexNormal
  vertex_normal_ = std::make_shared<VertexNormal>(mesh_);

  parameters = default_parameters();

  compute();
}

//-----------------------------------------------------------------------------
NodeNormal::~NodeNormal()
{
  // Enjoy your tea!
}


//-----------------------------------------------------------------------------
void NodeNormal::compute()
{
  Function Fnormal_cg1(V_CG1_);
  Function Ftau1_cg1(V_CG1_);
  Function Ftau2_cg1(V_CG1_);
  compute_CG1(Fnormal_cg1, Ftau1_cg1, Ftau2_cg1);

  if (V_ == V_CG1_)
  {
    *Fnormal_->vector() = *Fnormal_cg1.vector();
    *Ftau1_->vector() = *Ftau1_cg1.vector();
    *Ftau2_->vector() = *Ftau2_cg1.vector();
  }
  else
  {
    // Boundary
    class GeomBoundary: public SubDomain
    {
    public:
      bool inside(const Array<double> &x, bool on_boundary) const
      {
        return on_boundary;
      }
    };

    const std::size_t D = V_->mesh()->topology().dim();

    // Create a function with value 1 on the boundary
    auto boundary = std::make_shared<GeomBoundary>();
    std::vector<double> one_vector(D);
    std::fill(one_vector.begin(), one_vector.end(), 1.0);
    auto one = std::make_shared<Constant>(one_vector);
    DirichletBC dboundary(V_, one, boundary);
    Function u_one(V_);
    u_one.vector()->zero();
    dboundary.apply(*u_one.vector());

    // Interpolate to V and remove interior values
    Fnormal_->interpolate(Fnormal_cg1);
    Ftau1_->interpolate(Ftau1_cg1);
    Ftau2_->interpolate(Ftau2_cg1);
    *Fnormal_->vector() *= *u_one.vector();
    *Ftau1_->vector() *= *u_one.vector();
    *Ftau2_->vector() *= *u_one.vector();
  }

  // normalize
  normalize();
}

//-----------------------------------------------------------------------------
void NodeNormal::compute_CG1(Function& Fnormal, Function& Ftau1,
                             Function& Ftau2)
{
  Cell c(*mesh_, 0);
  const std::size_t D = mesh_->topology().dim();
  std::size_t local_dim = c.num_entities(0);
  std::vector<dolfin::la_index> id(D * local_dim);
  std::vector<double> normal_block(D * local_dim);
  std::vector<double> tau_block(D * local_dim);
  std::vector<double> tau1_block(D * local_dim);
  std::vector<double> tau2_block(D * local_dim);
  for (CellIterator cell(*mesh_); !cell.end(); ++cell)
  {
    auto cell_dofs =
      Fnormal.function_space()->dofmap()->cell_dofs((*cell).index());
    long int ii = 0;
    std::size_t jj = 0;

    for (std::size_t i = 0; i < D; ++i)
    {
      for (VertexIterator v(*cell); !v.end(); ++v, ++ii, ++jj)
      {
        normal_block[jj] = vertex_normal_->Fnormal[i][(*v).index()];
        tau1_block[jj] = vertex_normal_->Ftau1[i][(*v).index()];
        if (D == 3)
          tau2_block[jj] = vertex_normal_->Ftau2[i][(*v).index()];
        id[jj] = cell_dofs[ii];
      }
    }
    Fnormal.vector()->set_local(normal_block.data(), jj, id.data());
    Ftau1.vector()->set_local(tau1_block.data(), jj, id.data());
    Ftau2.vector()->set_local(tau2_block.data(), jj, id.data());
  }
  Fnormal.vector()->apply("insert");
  Ftau1.vector()->apply("insert");
  Ftau2.vector()->apply("insert");
}

//-----------------------------------------------------------------------------
void NodeNormal::normalize()
{
  const std::size_t D = mesh_->topology().dim();

  // Get the local values
  std::vector<double> n_values;
  std::vector<double> tau1_values;
  std::vector<double> tau2_values;
  Fnormal_->vector()->get_local(n_values);
  Ftau1_->vector()->get_local(tau1_values);
  if (D == 3)
    Ftau2_->vector()->get_local(tau2_values);

  const std::size_t num_sub_spaces = V_->element()->num_sub_elements();
  dolfin_assert(num_sub_spaces != 0);
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  std::vector<std::vector<dolfin::la_index>> dofs(num_sub_spaces);
  std::vector<std::size_t> dsizes(num_sub_spaces);
  for (std::size_t i = 0; i < num_sub_spaces; ++i)
  {
    dofs[i] = V_->sub(i)->dofmap()->dofs();
    dsizes[i] = dofs[i].size();
  }

  // All subspaces should have the same number of dofs in a vector space
  if (std::adjacent_find(dsizes.begin(), dsizes.end(),
                         std::not_equal_to<std::size_t>()) != dsizes.end())
  {
    dolfin::error("Different number of dofs for sub spaces");
  }

  const std::size_t dof_size = dsizes[0];
  auto mag_or_one = [](double mag) { return mag < DOLFIN_EPS ? 1.0 : mag; };

  // Normalize each node vector component-wise
  for (std::size_t i = 0; i < dof_size; ++i)
  {
    double mag_n = 0.0;
    double mag_tau1 = 0.0;
    double mag_tau2 = 0.0;
    for (std::size_t subspace = 0; subspace < num_sub_spaces; ++subspace)
    {
      const std::size_t dof_nr = dofs[subspace][i];
      mag_n += n_values[dof_nr - offset] * n_values[dof_nr - offset];
      mag_tau1 += tau1_values[dof_nr - offset] * tau1_values[dof_nr - offset];
      if (D == 3)
      {
        mag_tau2 += tau2_values[dof_nr - offset] *
                    tau2_values[dof_nr - offset];
      }
    }

    for (std::size_t subspace = 0; subspace < num_sub_spaces; ++subspace)
    {
      const std::size_t dof_nr = dofs[subspace][i];
      n_values[dof_nr - offset] /= mag_or_one(std::sqrt(mag_n));
      tau1_values[dof_nr - offset] /= mag_or_one(std::sqrt(mag_tau1));
      if (D == 3)
      {
        tau2_values[dof_nr - offset] /= mag_or_one(std::sqrt(mag_tau2));
      }
    }
  }

  Fnormal_->vector()->set_local(n_values);
  Ftau1_->vector()->set_local(tau1_values);
  Fnormal_->vector()->apply("insert");
  Ftau1_->vector()->apply("insert");
  if (D == 3)
  {
    Ftau2_->vector()->set_local(tau2_values);
    Ftau2_->vector()->apply("insert");
  }
}
  
void NodeNormal::normalize_vector(std::vector<double>& v,
                                      std::vector<double>& magnitudes){
  std::size_t local_dim = magnitudes.size();
  std::size_t D = v.size() / local_dim;
  for (std::size_t i = 0; i < local_dim; ++i){
    for (std::size_t j = 0; j < D; ++j){
      if (!dolfin::near(magnitudes[i], 0)){
        v[j*local_dim + i] /= magnitudes[i];
      }
    }
  }
}
