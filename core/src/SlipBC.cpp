// Copyright (C) 2007-2017 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Existing code for Dirichlet BC is used
//
// Modified by Niclas Jansson, 2008-2010.
// Modified by Lukas Lundgren 2021-2025.
// Modified by Christian Helanow, 2023.
//
// First added:  2007-05-01
// Last changed: 2017-02-18
/*
 Slip BC is implemented for cG(p) so far.
 Necessary variables are
 normal -   normal
 tau -      tangent in 2D
 tau_1 -    first tangent in 3D
 tau_2 -    second tangent in 3D

 nodetype - where the node lies:
 node_type = 1: node at the surface;
 node_type = 2: node at the edge;
 node_type = 3: node at the corner;

 n_type  - define type of node, in 3D: 1 - surface; 2 - edge; 3 - corner;
 in 2D: 1 - surface; 2 - corner;
 */

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/SparsityPatternBuilder.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/Progress.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/SubDomain.h>

///
using dolfin::FacetIterator;

#include "DofMappings.h"
#include "NodeNormal.h"
// #include <ocean/common/UnitNodeNormal.h>
#include "SlipBC.h"

using namespace fem_varden_flow;

//-----------------------------------------------------------------------------
SlipBC::SlipBC(std::shared_ptr<const FunctionSpace> V,
               std::shared_ptr<const SubDomain> sub_domain)
    : Hierarchical<SlipBC>(*this), V_(V), V_CG1_(V), mesh_(V->mesh()),
      user_sub_domain_(sub_domain) {
  init();
}

//-----------------------------------------------------------------------------
SlipBC::SlipBC(std::shared_ptr<const FunctionSpace> V,
               std::shared_ptr<const FunctionSpace> V_CG1,
               std::shared_ptr<const SubDomain> sub_domain)
    : Hierarchical<SlipBC>(*this), V_(V), V_CG1_(V_CG1), mesh_(V->mesh()),
      user_sub_domain_(sub_domain) {
  init();
}

//-----------------------------------------------------------------------------
SlipBC::SlipBC(std::shared_ptr<const FunctionSpace> V,
               std::shared_ptr<const FunctionSpace> V_CG1,
               std::shared_ptr<const MeshFunction<std::size_t>> sub_domains,
               std::size_t sub_domain)
    : Hierarchical<SlipBC>(*this), V_(V), V_CG1_(V_CG1), mesh_(V->mesh()),
      user_mesh_function_(sub_domains), user_sub_domain_marker_(sub_domain) {
  init();
}

//-----------------------------------------------------------------------------
SlipBC::~SlipBC() {
  // Do nothing
}

//-----------------------------------------------------------------------------
void SlipBC::init() {
  // construct normal functions
  Fnormal_ = std::make_shared<Function>(V_);
  Ftau1_ = std::make_shared<Function>(V_);
  Ftau2_ = std::make_shared<Function>(V_);

  if (user_sub_domain_) {
    init_from_sub_domain(user_sub_domain_);
  } else if (user_mesh_function_) {
    init_from_mesh_function(*user_mesh_function_, user_sub_domain_marker_);
  }

  compute_bc_topological();

  init_normals();

  // Init the sub-domain
  const std::size_t D = mesh_->topology().dim();
  sub_domains_ = std::make_shared<MeshFunction<std::size_t>>(mesh_, D - 1);

  parameters = default_parameters();
}

//-----------------------------------------------------------------------------
void SlipBC::init_normals() {
  const std::size_t D = mesh_->topology().dim();
  const std::size_t value_dim = V_->element()->value_dimension(0);

  std::shared_ptr<const FunctionSpace> V_normal;
  std::shared_ptr<const FunctionSpace> V_normal_CG1;
  if (D < value_dim) {
    V_normal = (V_->sub(0)->collapse());
    V_normal_CG1 = (V_CG1_->sub(0)->collapse());
  } else if (D == value_dim) {
    V_normal = V_;
    V_normal_CG1 = V_CG1_;
  }

  NodeNormal node_normal(V_normal, V_normal_CG1);
  // UnitNodeNormal node_normal(V_normal, V_normal_CG1, true);

  // File fnormal("results/normal.pvd", "compressed");
  // fnormal << node_normal.normal();
  // fnormal << node_normal.tau1();
  // fnormal << node_normal.tau2();

  // Assign the normal vectors to space W
  std::vector<dolfin::la_index> dof_W;
  std::vector<dolfin::la_index> dof_V;
  DofMappings dofmappings;
  dofmappings.get_all_dofs(*Fnormal_, dof_W);
  dofmappings.get_all_dofs(node_normal.normal(), dof_V);
  std::vector<double> Fn_arr(dof_V.size());
  std::vector<double> Ft1_arr(dof_V.size());
  std::vector<double> Ft2_arr(dof_V.size());
  std::vector<double> Fwn_arr(dof_W.size());
  std::vector<double> Fwt1_arr(dof_W.size());
  std::vector<double> Fwt2_arr(dof_W.size());

  node_normal.normal().vector()->get_local(&Fn_arr[0], dof_V.size(), &dof_V[0]);
  node_normal.tau1().vector()->get_local(&Ft1_arr[0], dof_V.size(), &dof_V[0]);
  if (D == 3) {
    node_normal.tau2().vector()->get_local(&Ft2_arr[0], dof_V.size(),
                                           &dof_V[0]);
  }

  std::size_t num_cell_nodes =
      Fnormal_->function_space()->dofmap()->max_element_dofs() /
      Fnormal_->value_size();

  std::size_t offset_W = 0;
  std::size_t offset_M = 0;
  for (CellIterator cell(*mesh_); !cell.end(); ++cell) {
    for (std::size_t node = 0; node < num_cell_nodes; ++node) {
      for (int j = 0; j < D; ++j) {
        std::size_t mj_pos_w = node + offset_W + num_cell_nodes * j;
        std::size_t mj_pos_m = node + offset_M + num_cell_nodes * j;
        Fwn_arr[mj_pos_w] = Fn_arr[mj_pos_m];
        Fwt1_arr[mj_pos_w] = Ft1_arr[mj_pos_m];
        if (D == 3) {
          Fwt2_arr[mj_pos_w] = Ft2_arr[mj_pos_m];
        }
      }
    }
    offset_W += V_->dofmap()->max_element_dofs();
    offset_M += V_normal->dofmap()->max_element_dofs();
  }
  Fnormal_->vector()->set_local(&Fwn_arr[0], dof_W.size(), &dof_W[0]);
  Ftau1_->vector()->set_local(&Fwt1_arr[0], dof_W.size(), &dof_W[0]);
  Fnormal_->vector()->apply("insert");
  Ftau1_->vector()->apply("insert");
  if (D == 3) {
    Ftau2_->vector()->set_local(&Fwt2_arr[0], dof_W.size(), &dof_W[0]);
    Ftau2_->vector()->apply("insert");
  }
}

//-----------------------------------------------------------------------------
void SlipBC::init_from_sub_domain(
    std::shared_ptr<const SubDomain> sub_domain) const {
  dolfin_assert(facets_.empty());

  // Create mesh function for sub domain markers on facets and mark
  // all facet as subdomain 1
  const std::size_t dim = mesh_->topology().dim();
  mesh_->init(dim - 1);
  MeshFunction<std::size_t> sub_domains(mesh_, mesh_->topology().dim() - 1, 1);

  // Mark the sub domain as sub domain 0
  sub_domain->mark(sub_domains, 0, "true");

  // Initialize from mesh function
  init_from_mesh_function(sub_domains, 0);
}

//-----------------------------------------------------------------------------
void SlipBC::init_from_mesh_function(
    const MeshFunction<std::size_t> &sub_domains,
    std::size_t sub_domain) const {
  // Make sure we have the facet - cell connectivity
  const std::size_t D = mesh_->topology().dim();
  mesh_->init(D - 1, D);

  // Build set of boundary facets
  dolfin_assert(facets_.empty());
  for (FacetIterator facet(*mesh_); !facet.end(); ++facet) {
    if (sub_domains[*facet] == sub_domain)
      facets_.push_back(facet->index());
  }
}

//-----------------------------------------------------------------------------
void SlipBC::init_facets(const MPI_Comm mpi_comm) const {
  Timer timer("DirichletBC init facets");

  if (dolfin::MPI::max(mpi_comm, facets_.size()) > 0)
    return;

  if (user_sub_domain_) {
    init_from_sub_domain(user_sub_domain_);
  }
}

//-----------------------------------------------------------------------------
void SlipBC::compute_bc_topological() {
  dolfin_assert(V_);
  LocalData data(*V_);

  // Extract the list of facets where the BC should be applied
  init_facets(mesh_->mpi_comm());

  // Special case
  if (facets_.empty()) {
    if (dolfin::MPI::size(mesh_->mpi_comm()) == 1) {
      warning("Found no facets matching domain for boundary condition.");
    }
    return;
  }

  // Get dofmap
  dolfin_assert(V_->dofmap());
  const GenericDofMap &dofmap = *V_->dofmap();

  // Get finite element
  dolfin_assert(V_->element());
  const FiniteElement &element = *V_->element();

  // Topological dimension
  const std::size_t D = mesh_->topology().dim();

  // Initialise facet-cell connectivity
  mesh_->init(D);
  mesh_->init(D - 1, D);

  // Create UFC cell
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;

  // Iterate over marked
  Progress p("Computing Dirichlet boundary values, topological search",
             facets_.size());
  std::size_t slip_dofs_size = facets_.size() * dofmap.num_facet_dofs();

  std::size_t value_dim = V_->element()->value_dimension(0);

  std::size_t num_cell_nodes =
      V_->dofmap()->max_element_dofs() / V_->element()->value_dimension(0);

  std::size_t ii = 0;
  for (std::size_t f = 0; f < facets_.size(); ++f) {
    // Create facet
    const Facet facet(*mesh_, facets_[f]);

    // Get cell to which facet belongs.
    dolfin_assert(facet.num_entities(D) > 0);
    const std::size_t cell_index = facet.entities(D)[0];

    // Create attached cell
    const Cell cell(*mesh_, cell_index);

    // Get local index of facet with respect to the cell
    const size_t facet_local_index = cell.index(facet);

    // Update UFC cell geometry data
    cell.get_vertex_coordinates(vertex_coordinates);
    cell.get_cell_data(ufc_cell, facet_local_index);

    // Tabulate dofs on cell
    auto cell_dofs = dofmap.cell_dofs(cell.index());

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, facet_local_index);

    std::size_t offset_v = dofmap.max_element_dofs();
    // Pick values for facet
    for (std::size_t i = 0; i < data.facet_dofs.size() / value_dim; ++i) {
      std::vector<std::size_t> dofs(D);
      for (std::size_t j = 0; j < D; ++j) {
        dofs[j] = cell_dofs[data.facet_dofs[i] + num_cell_nodes * j];
      }
      v_slip_dofs_.push_back(dofs);
    }
    p++;
  }
  //   Erase repeated dofs
  sort(v_slip_dofs_.begin(), v_slip_dofs_.end());
  v_slip_dofs_.erase(unique(v_slip_dofs_.begin(), v_slip_dofs_.end()),
                     v_slip_dofs_.end());
}

//-----------------------------------------------------------------------------
void SlipBC::apply(GenericMatrix &A, GenericVector &b) {
  const std::size_t D = mesh_->topology().dim();

  As_ = std::make_shared<Matrix>(A);
  bs_ = std::make_shared<Vector>(b);

  for (std::size_t i = 0; i < v_slip_dofs_.size(); ++i) {
    std::vector<std::size_t> dofs(D);
    dofs = v_slip_dofs_[i];
    applySlipBC(A, b, dofs);
  }

  b.apply("insert");
  A.apply("insert");
}

//-----------------------------------------------------------------------------
void SlipBC::applySlipBC(GenericMatrix &A, GenericVector &b,
                         std::vector<std::size_t> &dofs) {
  // Get the number of space dimensions of the problem
  const std::size_t D = mesh_->topology().dim();
  std::pair<std::size_t, std::size_t> rows = As_->local_range(0);

  std::pair<std::size_t, std::size_t> ownership_range =
      V_->dofmap()->ownership_range();
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  for (std::size_t i = 0; i < dofs.size(); ++i) {
    if (!(dofs[i] + offset >= rows.first && dofs[i] + offset < rows.second)) {
      return;
    }
  }

  std::vector<std::vector<double>> a(D);
  std::vector<std::vector<std::size_t>> a_indices(D);
  std::vector<double> l(D);
  std::vector<double> n(D);
  std::vector<double> t1(D);
  std::vector<double> t2(D);

  double max_n = 0.0;
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    As_->getrow(dofs[i] + offset, a_indices[i], a[i]);
    l[i] = (*bs_)[dofs[i]];
    n[i] = (*Fnormal_->vector())[dofs[i]];
    t1[i] = (*Ftau1_->vector())[dofs[i]];
    if (D == 3) {
      t2[i] = (*Ftau2_->vector())[dofs[i]];
    }
    max_n = std::max(max_n, fabs(n[i]));
  }
  std::size_t nb_cols = a_indices[0].size();

  // Find the largest |n_i| and put it to the diagonal
  std::vector<std::size_t> row(D);
  if (fabs(max_n - fabs(n[0])) < DOLFIN_EPS) {
    row[0] = 0;
    row[1] = 1;
    if (D == 3) {
      row[2] = 2;
    }
  } else if (fabs(max_n - fabs(n[1])) < DOLFIN_EPS) {
    row[0] = 1;
    row[1] = 0;
    if (D == 3) {
      row[2] = 2;
    }
  } else {
    if (D == 3) {
      row[0] = 2;
      row[1] = 1;
      row[2] = 0;
    }
  }

  // Set the row for normal
  std::vector<double> zero_block(nb_cols);
  std::fill(zero_block.begin(), zero_block.end(), 0.0);

  A.setrow(dofs[row[0]] + offset, a_indices[row[0]], zero_block);
  for (std::size_t i = 0; i < D; ++i) {
    Aset(A, dofs[row[0]] + offset, dofs[i] + offset, n[i]);
  }
  bset(b, dofs[row[0]], 0.0);

  // compute the row with tau1
  std::vector<double> a_tau1(nb_cols);
  for (std::size_t i = 0; i < nb_cols; ++i) {
    a_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a_tau1[i] += a[j][i] * t1[j];
    }
  }
  double l_tau1 = 0.0;
  for (std::size_t j = 0; j < D; ++j) {
    l_tau1 += l[j] * t1[j];
  }

  // set the row for tau1
  if (D == 2) {
    A.setrow(dofs[row[1]] + offset, a_indices[row[1]], a_tau1);
    bset(b, dofs[row[1]], l_tau1);
  } else if (D == 3) {
    // compute the row with tau2
    std::vector<double> a_tau2(nb_cols);
    for (std::size_t i = 0; i < nb_cols; ++i) {
      a_tau2[i] = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        a_tau2[i] += a[j][i] * t2[j];
      }
    }
    double l_tau2 = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      l_tau2 += l[j] * t2[j];
    }

    std::size_t ind1 = 0;
    std::size_t ind2 = 0;
    for (std::size_t i = 0; i < nb_cols; ++i) {
      if (dofs[row[1]] + offset == a_indices[row[1]][i])
        ind1 = i;
      if (dofs[row[2]] + offset == a_indices[row[1]][i])
        ind2 = i;
    }

    // find maximum of the elements of 2 vectors which are belong to diagonal
    double row_d11 =
        a[0][ind1] * t1[0] + a[1][ind1] * t1[1] + a[2][ind1] * t1[2];
    double row_d21 =
        a[0][ind1] * t2[0] + a[1][ind1] * t2[1] + a[2][ind1] * t2[2];
    double row_d12 =
        a[0][ind2] * t1[0] + a[1][ind2] * t1[1] + a[2][ind2] * t1[2];
    double row_d22 =
        a[0][ind2] * t2[0] + a[1][ind2] * t2[1] + a[2][ind2] * t2[2];

    double maxr = std::max(std::fabs(row_d11), std::fabs(row_d21));
    maxr = std::max(maxr, std::fabs(row_d12));
    maxr = std::max(maxr, std::fabs(row_d22));

    std::size_t r1 = row[1];
    std::size_t r2 = row[2];
    std::size_t row_d1 = 0;
    std::size_t row_d2 = 0;
    if ((std::fabs(std::fabs(row_d11) - maxr) < DOLFIN_EPS) ||
        (std::fabs(std::fabs(row_d22) - maxr) < DOLFIN_EPS)) {
      row_d1 = r1;
      row_d2 = r2;
    } else {
      row_d1 = r2;
      row_d2 = r1;
    }

    A.setrow(dofs[row_d1] + offset, a_indices[row_d1], a_tau1);
    bset(b, dofs[row_d1], l_tau1);
    A.setrow(dofs[row_d2] + offset, a_indices[row_d2], a_tau2);
    bset(b, dofs[row_d2], l_tau2);
  }
}

//-----------------------------------------------------------------------------
SlipBC::LocalData::LocalData(const FunctionSpace &V)
    : w(V.dofmap()->max_element_dofs(), 0.0),
      facet_dofs(V.dofmap()->num_facet_dofs(), 0),
      coordinates(boost::extents[V.dofmap()->max_element_dofs()]
                                [V.mesh()->geometry().dim()]) {
  // Do nothing
}

//-----------------------------------------------------------------------------
void SlipBC::applyZeroSlipBC(GenericMatrix &A, std::vector<std::size_t> &dofs) {
  // Get the number of space dimensions of the problem
  const std::size_t D = mesh_->topology().dim();
  std::pair<std::size_t, std::size_t> rows = As_zero_->local_range(0);

  std::pair<std::size_t, std::size_t> ownership_range =
      V_->dofmap()->ownership_range();
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  for (std::size_t i = 0; i < dofs.size(); ++i) {
    if (!(dofs[i] + offset >= rows.first && dofs[i] + offset < rows.second)) {
      return;
    }
  }

  std::vector<std::vector<double>> a(D);
  std::vector<std::vector<std::size_t>> a_indices(D);
  std::vector<double> l(D);
  std::vector<double> n(D);
  std::vector<double> zeros(D);
  std::fill(zeros.begin(), zeros.end(), 0.0);
  std::vector<double> t1(D);
  std::vector<double> t2(D);

  double max_n = 0.0;
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    As_zero_->getrow(dofs[i] + offset, a_indices[i], a[i]);
    // l[i] = (*bs_)[dofs[i]];
    n[i] = (*Fnormal_->vector())[dofs[i]];
    t1[i] = (*Ftau1_->vector())[dofs[i]];
    if (D == 3) {
      t2[i] = (*Ftau2_->vector())[dofs[i]];
    }
    max_n = std::max(max_n, fabs(n[i]));
  }
  std::size_t nb_cols = a_indices[0].size();

  // Find the largest |n_i| and put it to the diagonal
  std::vector<std::size_t> row(D);
  if (fabs(max_n - fabs(n[0])) < DOLFIN_EPS) {
    row[0] = 0;
    row[1] = 1;
    if (D == 3) {
      row[2] = 2;
    }
  } else if (fabs(max_n - fabs(n[1])) < DOLFIN_EPS) {
    row[0] = 1;
    row[1] = 0;
    if (D == 3) {
      row[2] = 2;
    }
  } else {
    if (D == 3) {
      row[0] = 2;
      row[1] = 1;
      row[2] = 0;
    }
  }
  // Set the row for normal
  std::vector<double> zero_block(nb_cols);
  std::fill(zero_block.begin(), zero_block.end(), 0.0);

  A.setrow(dofs[row[0]] + offset, a_indices[row[0]], zero_block);

  // not neccessary????
  /*  for (std::size_t i = 0; i < D; ++i)
    {
      Aset(A, dofs[row[0]] + offset, dofs[i] + offset, n[i]);
    }*/

  // compute the row with tau1
  std::vector<double> a_tau1(nb_cols);
  for (std::size_t i = 0; i < nb_cols; ++i) {
    a_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a_tau1[i] += a[j][i] * t1[j];
    }
  }
  double l_tau1 = 0.0;
  for (std::size_t j = 0; j < D; ++j) {
    // l_tau1 += l[j] * t1[j];
  }

  // set the row for tau1
  if (D == 2) {
    A.setrow(dofs[row[1]] + offset, a_indices[row[1]], a_tau1);
  } else if (D == 3) // need to test if D==3 works!
  {
    // compute the row with tau2
    std::vector<double> a_tau2(nb_cols);
    for (std::size_t i = 0; i < nb_cols; ++i) {
      a_tau2[i] = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        a_tau2[i] += a[j][i] * t2[j];
      }
    }
    double l_tau2 = 0.0; // only neccessary for rhs so we ignore it
    for (std::size_t j = 0; j < D; ++j) {
      // l_tau2 += l[j] * t2[j];
    }

    std::size_t ind1 = 0;
    std::size_t ind2 = 0;
    for (std::size_t i = 0; i < nb_cols; ++i) {
      if (dofs[row[1]] + offset == a_indices[row[1]][i])
        ind1 = i;
      if (dofs[row[2]] + offset == a_indices[row[1]][i])
        ind2 = i;
    }

    // find maximum of the elements of 2 vectors which are belong to diagonal
    double row_d11 =
        a[0][ind1] * t1[0] + a[1][ind1] * t1[1] + a[2][ind1] * t1[2];
    double row_d21 =
        a[0][ind1] * t2[0] + a[1][ind1] * t2[1] + a[2][ind1] * t2[2];
    double row_d12 =
        a[0][ind2] * t1[0] + a[1][ind2] * t1[1] + a[2][ind2] * t1[2];
    double row_d22 =
        a[0][ind2] * t2[0] + a[1][ind2] * t2[1] + a[2][ind2] * t2[2];

    double maxr = std::max(std::fabs(row_d11), std::fabs(row_d21));
    maxr = std::max(maxr, std::fabs(row_d12));
    maxr = std::max(maxr, std::fabs(row_d22));

    std::size_t r1 = row[1];
    std::size_t r2 = row[2];
    std::size_t row_d1 = 0;
    std::size_t row_d2 = 0;
    if ((std::fabs(std::fabs(row_d11) - maxr) < DOLFIN_EPS) ||
        (std::fabs(std::fabs(row_d22) - maxr) < DOLFIN_EPS)) {
      row_d1 = r1;
      row_d2 = r2;
    } else {
      row_d1 = r2;
      row_d2 = r1;
    }

    A.setrow(dofs[row_d1] + offset, a_indices[row_d1], a_tau1);
    // bset(b, dofs[row_d1], l_tau1);
    A.setrow(dofs[row_d2] + offset, a_indices[row_d2], a_tau2);
    // bset(b, dofs[row_d2], l_tau2);
  }

  // had this before
  // A.setrow(dofs[row[1]] + offset, a_indices[row[1]], zero_block);
}

//-----------------------------------------------------------------------------
void SlipBC::zero(GenericMatrix &A) {
  const std::size_t D = mesh_->topology().dim();

  As_zero_ = std::make_shared<Matrix>(A);

  for (std::size_t i = 0; i < v_slip_dofs_.size(); ++i) {
    std::vector<std::size_t> dofs(D);
    dofs = v_slip_dofs_[i];
    applyZeroSlipBC(A, dofs);
  }
  A.apply("insert");
}

//-----------------------------------------------------------------------------
void SlipBC::apply_system(GenericMatrix &A, GenericMatrix &A01,
                          GenericVector &b) {
  const std::size_t D = mesh_->topology().dim();

  As_ = std::make_shared<Matrix>(A);
  As_zero_ = std::make_shared<Matrix>(A01);
  bs_ = std::make_shared<Vector>(b);

  for (std::size_t i = 0; i < v_slip_dofs_.size(); ++i) {
    std::vector<std::size_t> dofs(D);
    dofs = v_slip_dofs_[i];
    applySystemSlipBC(A, A01, b, dofs);
  }

  b.apply("insert");
  A.apply("insert");
  A01.apply("insert");
}

//-----------------------------------------------------------------------------
void SlipBC::applySystemSlipBC(GenericMatrix &A, GenericMatrix &A01,
                               GenericVector &b,
                               std::vector<std::size_t> &dofs) {
  // Get the number of space dimensions of the problem
  const std::size_t D = mesh_->topology().dim();
  std::pair<std::size_t, std::size_t> rows = As_->local_range(0);
  std::pair<std::size_t, std::size_t> rows01 = As_zero_->local_range(0);

  std::pair<std::size_t, std::size_t> ownership_range =
      V_->dofmap()->ownership_range();
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  for (std::size_t i = 0; i < dofs.size(); ++i) {
    if (!(dofs[i] + offset >= rows.first && dofs[i] + offset < rows.second)) {
      return;
    }
  }

  std::vector<std::vector<double>> a(D);
  std::vector<std::vector<double>> a01(D);
  std::vector<std::vector<std::size_t>> a_indices(D);
  std::vector<std::vector<std::size_t>> a01_indices(D);
  std::vector<double> l(D);
  std::vector<double> n(D);
  std::vector<double> zeros(D);
  std::fill(zeros.begin(), zeros.end(), 0.0);
  std::vector<double> t1(D);
  std::vector<double> t2(D);

  double max_n = 0.0;
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    As_->getrow(dofs[i] + offset, a_indices[i], a[i]);
    As_zero_->getrow(dofs[i] + offset, a01_indices[i], a01[i]);
    l[i] = (*bs_)[dofs[i]];
    n[i] = (*Fnormal_->vector())[dofs[i]];
    t1[i] = (*Ftau1_->vector())[dofs[i]];
    if (D == 3) {
      t2[i] = (*Ftau2_->vector())[dofs[i]];
    }
    max_n = std::max(max_n, fabs(n[i]));
  }
  std::size_t nb_cols = a_indices[0].size();
  std::size_t nb_cols_01 = a01_indices[0].size();

  // Find the largest |n_i| and put it to the diagonal
  std::vector<std::size_t> row(D);
  if (fabs(max_n - fabs(n[0])) < DOLFIN_EPS) {
    row[0] = 0;
    row[1] = 1;
    if (D == 3) {
      row[2] = 2;
    }
  } else if (fabs(max_n - fabs(n[1])) < DOLFIN_EPS) {
    row[0] = 1;
    row[1] = 0;
    if (D == 3) {
      row[2] = 2;
    }
  } else {
    if (D == 3) {
      row[0] = 2;
      row[1] = 1;
      row[2] = 0;
    }
  }
  // Set the row for normal
  std::vector<double> zero_block(nb_cols);
  std::fill(zero_block.begin(), zero_block.end(), 0.0);

  std::vector<double> zero_block_01(nb_cols_01);
  std::fill(zero_block_01.begin(), zero_block_01.end(), 0.0);

  A.setrow(dofs[row[0]] + offset, a_indices[row[0]], zero_block);
  A01.setrow(dofs[row[0]] + offset, a01_indices[row[0]], zero_block_01);

  for (std::size_t i = 0; i < D; ++i) {
    Aset(A, dofs[row[0]] + offset, dofs[i] + offset, n[i]);
  }
  bset(b, dofs[row[0]], 0.0);

  // compute the row with tau1
  std::vector<double> a_tau1(nb_cols);
  for (std::size_t i = 0; i < nb_cols; ++i) {
    a_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a_tau1[i] += a[j][i] * t1[j];
    }
  }

  // compute the row with tau1
  std::vector<double> a01_tau1(nb_cols_01);
  for (std::size_t i = 0; i < nb_cols_01; ++i) {
    a01_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a01_tau1[i] += a01[j][i] * t1[j];
    }
  }

  double l_tau1 = 0.0;
  for (std::size_t j = 0; j < D; ++j) {
    l_tau1 += l[j] * t1[j];
  }

  // set the row for tau1
  if (D == 2) {
    A.setrow(dofs[row[1]] + offset, a_indices[row[1]], a_tau1);
    A01.setrow(dofs[row[1]] + offset, a01_indices[row[1]], a01_tau1);
    bset(b, dofs[row[1]], l_tau1);
  } else if (D == 3) {
    // compute the row with tau2
    std::vector<double> a_tau2(nb_cols);
    for (std::size_t i = 0; i < nb_cols; ++i) {
      a_tau2[i] = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        a_tau2[i] += a[j][i] * t2[j];
      }
    }
    std::vector<double> a01_tau2(nb_cols_01);
    for (std::size_t i = 0; i < nb_cols_01; ++i) {
      a01_tau2[i] = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        a01_tau2[i] += a01[j][i] * t2[j];
      }
    }
    double l_tau2 = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      l_tau2 += l[j] * t2[j];
    }

    std::size_t ind1 = 0;
    std::size_t ind2 = 0;
    for (std::size_t i = 0; i < nb_cols; ++i) {
      if (dofs[row[1]] + offset == a_indices[row[1]][i])
        ind1 = i;
      if (dofs[row[2]] + offset == a_indices[row[1]][i])
        ind2 = i;
    }

    // find maximum of the elements of 2 vectors which are belong to diagonal
    double row_d11 =
        a[0][ind1] * t1[0] + a[1][ind1] * t1[1] + a[2][ind1] * t1[2];
    double row_d21 =
        a[0][ind1] * t2[0] + a[1][ind1] * t2[1] + a[2][ind1] * t2[2];
    double row_d12 =
        a[0][ind2] * t1[0] + a[1][ind2] * t1[1] + a[2][ind2] * t1[2];
    double row_d22 =
        a[0][ind2] * t2[0] + a[1][ind2] * t2[1] + a[2][ind2] * t2[2];

    double maxr = std::max(std::fabs(row_d11), std::fabs(row_d21));
    maxr = std::max(maxr, std::fabs(row_d12));
    maxr = std::max(maxr, std::fabs(row_d22));

    std::size_t r1 = row[1];
    std::size_t r2 = row[2];
    std::size_t row_d1 = 0;
    std::size_t row_d2 = 0;
    if ((std::fabs(std::fabs(row_d11) - maxr) < DOLFIN_EPS) ||
        (std::fabs(std::fabs(row_d22) - maxr) < DOLFIN_EPS)) {
      row_d1 = r1;
      row_d2 = r2;
    } else {
      row_d1 = r2;
      row_d2 = r1;
    }

    A.setrow(dofs[row_d1] + offset, a_indices[row_d1], a_tau1);
    A01.setrow(dofs[row_d1] + offset, a01_indices[row_d1], a01_tau1);
    bset(b, dofs[row_d1], l_tau1);
    A.setrow(dofs[row_d2] + offset, a_indices[row_d2], a_tau2);
    A01.setrow(dofs[row_d2] + offset, a01_indices[row_d2], a01_tau2);
    bset(b, dofs[row_d2], l_tau2);
  }
}

//-----------------------------------------------------------------------------
void SlipBC::apply_inflow(GenericMatrix &A, GenericVector &b,
                          GenericVector &g) {
  const std::size_t D = mesh_->topology().dim();

  As_ = std::make_shared<Matrix>(A);
  bs_ = std::make_shared<Vector>(b);

  for (std::size_t i = 0; i < v_slip_dofs_.size(); ++i) {
    std::vector<std::size_t> dofs(D);
    dofs = v_slip_dofs_[i];
    applyInflowBC(A, b, dofs, g);
  }

  b.apply("insert");
  A.apply("insert");
}

//-----------------------------------------------------------------------------
void SlipBC::apply_inflow_system(GenericMatrix &A, GenericMatrix &A01,
                                 GenericVector &b, GenericVector &g) {
  const std::size_t D = mesh_->topology().dim();

  As_ = std::make_shared<Matrix>(A);
  As_zero_ = std::make_shared<Matrix>(A01);
  bs_ = std::make_shared<Vector>(b);

  for (std::size_t i = 0; i < v_slip_dofs_.size(); ++i) {
    std::vector<std::size_t> dofs(D);
    dofs = v_slip_dofs_[i];
    applySystemInflowBC(A, A01, b, dofs, g);
  }

  b.apply("insert");
  A.apply("insert");
  A01.apply("insert");
}

// Lukas: I only fixed system version so that it works. This one still needs to
// be implemented properly.
//-----------------------------------------------------------------------------
void SlipBC::applyInflowBC(GenericMatrix &A, GenericVector &b,
                           std::vector<std::size_t> &dofs, GenericVector &g) {
  // Get the number of space dimensions of the problem
  const std::size_t D = mesh_->topology().dim();
  std::pair<std::size_t, std::size_t> rows = As_->local_range(0);

  std::pair<std::size_t, std::size_t> ownership_range =
      V_->dofmap()->ownership_range();
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  for (std::size_t i = 0; i < dofs.size(); ++i) {
    if (!(dofs[i] + offset >= rows.first && dofs[i] + offset < rows.second)) {
      return;
    }
  }

  std::vector<std::vector<double>> a(D);
  std::vector<std::vector<std::size_t>> a_indices(D);
  std::vector<double> l(D);
  std::vector<double> n(D);
  std::vector<double> t1(D);
  std::vector<double> t2(D);

  double max_n = 0.0;
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    As_->getrow(dofs[i] + offset, a_indices[i], a[i]);
    l[i] = (*bs_)[dofs[i]];
    n[i] = (*Fnormal_->vector())[dofs[i]];
    t1[i] = (*Ftau1_->vector())[dofs[i]];
    if (D == 3) {
      t2[i] = (*Ftau2_->vector())[dofs[i]];
    }
    max_n = std::max(max_n, fabs(n[i]));
  }
  std::size_t nb_cols = a_indices[0].size();

  // Find the largest |n_i| and put it to the diagonal
  std::vector<std::size_t> row(D);
  if (fabs(max_n - fabs(n[0])) < DOLFIN_EPS) {
    row[0] = 0;
    row[1] = 1;
    if (D == 3) {
      row[2] = 2;
    }
  } else if (fabs(max_n - fabs(n[1])) < DOLFIN_EPS) {
    row[0] = 1;
    row[1] = 0;
    if (D == 3) {
      row[2] = 2;
    }
  } else {
    if (D == 3) {
      row[0] = 2;
      row[1] = 1;
      row[2] = 0;
    }
  }

  // Set the row for normal
  std::vector<double> zero_block(nb_cols);
  std::fill(zero_block.begin(), zero_block.end(), 0.0);

  A.setrow(dofs[row[0]] + offset, a_indices[row[0]], zero_block);
  double g_value = 0.0;
  bget(g, dofs[row[0]], g_value);
  for (std::size_t i = 0; i < D; ++i) {
    Aset(A, dofs[row[0]] + offset, dofs[i] + offset, n[i]);
  }
  bset(b, dofs[row[0]], g_value); // set to g instead of 0.0

  // compute the row with tau1
  std::vector<double> a_tau1(nb_cols);
  for (std::size_t i = 0; i < nb_cols; ++i) {
    a_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a_tau1[i] += t1[j]; // before: a[j][i] * t1[j];
    }
  }
  double l_tau1 = 0.0;
  for (std::size_t j = 0; j < D; ++j) {
    l_tau1 += 0.0; // before: l[j] * t1[j];
  }

  // set the row for tau1
  if (D == 2) {
    A.setrow(dofs[row[1]] + offset, a_indices[row[1]], a_tau1);
    bset(b, dofs[row[1]], 0.0); // before: l_tau1
  } else if (D == 3) {
    // compute the row with tau2
    std::vector<double> a_tau2(nb_cols);
    for (std::size_t i = 0; i < nb_cols; ++i) {
      a_tau2[i] = 0.0;
      for (std::size_t j = 0; j < D; ++j) {
        a_tau2[i] += t2[j]; // before: a[j][i] * t2[j]
      }
    }
    double l_tau2 = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      l_tau2 += 0.0; // before: l[j] * t2[j]
    }

    std::size_t ind1 = 0;
    std::size_t ind2 = 0;
    for (std::size_t i = 0; i < nb_cols; ++i) {
      if (dofs[row[1]] + offset == a_indices[row[1]][i])
        ind1 = i;
      if (dofs[row[2]] + offset == a_indices[row[1]][i])
        ind2 = i;
    }

    // find maximum of the elements of 2 vectors which are belong to diagonal
    double row_d11 =
        a[0][ind1] * t1[0] + a[1][ind1] * t1[1] + a[2][ind1] * t1[2];
    double row_d21 =
        a[0][ind1] * t2[0] + a[1][ind1] * t2[1] + a[2][ind1] * t2[2];
    double row_d12 =
        a[0][ind2] * t1[0] + a[1][ind2] * t1[1] + a[2][ind2] * t1[2];
    double row_d22 =
        a[0][ind2] * t2[0] + a[1][ind2] * t2[1] + a[2][ind2] * t2[2];

    double maxr = std::max(std::fabs(row_d11), std::fabs(row_d21));
    maxr = std::max(maxr, std::fabs(row_d12));
    maxr = std::max(maxr, std::fabs(row_d22));

    std::size_t r1 = row[1];
    std::size_t r2 = row[2];
    std::size_t row_d1 = 0;
    std::size_t row_d2 = 0;
    if ((std::fabs(std::fabs(row_d11) - maxr) < DOLFIN_EPS) ||
        (std::fabs(std::fabs(row_d22) - maxr) < DOLFIN_EPS)) {
      row_d1 = r1;
      row_d2 = r2;
    } else {
      row_d1 = r2;
      row_d2 = r1;
    }

    A.setrow(dofs[row_d1] + offset, a_indices[row_d1], a_tau1);
    bset(b, dofs[row_d1], l_tau1);
    A.setrow(dofs[row_d2] + offset, a_indices[row_d2], a_tau2);
    bset(b, dofs[row_d2], l_tau2);
  }
}

//-----------------------------------------------------------------------------
void SlipBC::applySystemInflowBC(GenericMatrix &A, GenericMatrix &A01,
                                 GenericVector &b,
                                 std::vector<std::size_t> &dofs,
                                 GenericVector &g) {
  // Get the number of space dimensions of the problem
  const std::size_t D = mesh_->topology().dim();
  std::pair<std::size_t, std::size_t> rows = As_->local_range(0);
  std::pair<std::size_t, std::size_t> rows01 = As_zero_->local_range(0);

  std::pair<std::size_t, std::size_t> ownership_range =
      V_->dofmap()->ownership_range();
  const std::size_t offset = V_->dofmap()->ownership_range().first;

  for (std::size_t i = 0; i < dofs.size(); ++i) {
    if (!(dofs[i] + offset >= rows.first && dofs[i] + offset < rows.second)) {
      return;
    }
  }

  std::vector<std::vector<double>> a(D);
  std::vector<std::vector<double>> a01(D);
  std::vector<std::vector<std::size_t>> a_indices(D);
  std::vector<std::vector<std::size_t>> a01_indices(D);
  std::vector<double> l(D);
  std::vector<double> n(D);
  std::vector<double> zeros(D);
  std::fill(zeros.begin(), zeros.end(), 0.0);
  std::vector<double> t1(D);
  std::vector<double> t2(D);

  double max_n = 0.0;
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    As_->getrow(dofs[i] + offset, a_indices[i], a[i]);
    As_zero_->getrow(dofs[i] + offset, a01_indices[i], a01[i]);
    l[i] = (*bs_)[dofs[i]];
    n[i] = (*Fnormal_->vector())[dofs[i]];
    t1[i] = (*Ftau1_->vector())[dofs[i]];
    if (D == 3) {
      t2[i] = (*Ftau2_->vector())[dofs[i]];
    }
    max_n = std::max(max_n, fabs(n[i]));
  }
  std::size_t nb_cols = a_indices[0].size();
  std::size_t nb_cols_01 = a01_indices[0].size();

  // Find the largest |n_i| and put it to the diagonal
  std::vector<std::size_t> row(D);
  if (fabs(max_n - fabs(n[0])) < DOLFIN_EPS) {
    row[0] = 0;
    row[1] = 1;
    if (D == 3) {
      row[2] = 2;
    }
  } else if (fabs(max_n - fabs(n[1])) < DOLFIN_EPS) {
    row[0] = 1;
    row[1] = 0;
    if (D == 3) {
      row[2] = 2;
    }
  } else {
    if (D == 3) {
      row[0] = 2;
      row[1] = 1;
      row[2] = 0;
    }
  }

  // Set the row for normal
  std::vector<double> zero_block(nb_cols);
  std::fill(zero_block.begin(), zero_block.end(), 0.0);

  std::vector<double> zero_block_01(nb_cols_01);
  std::fill(zero_block_01.begin(), zero_block_01.end(), 0.0);

  A.setrow(dofs[row[0]] + offset, a_indices[row[0]], zero_block);
  A.setrow(dofs[row[1]] + offset, a_indices[row[1]], zero_block); // new

  A01.setrow(dofs[row[0]] + offset, a01_indices[row[0]], zero_block_01);
  A01.setrow(dofs[row[1]] + offset, a01_indices[row[1]], zero_block_01); // new

  if (D == 3) {
    A.setrow(dofs[row[2]] + offset, a_indices[row[2]], zero_block); // new
    A01.setrow(dofs[row[2]] + offset, a01_indices[row[2]],
               zero_block_01); // new
  }

  double g_value = 0.0;
  bget(g, dofs[row[0]], g_value);
  for (std::size_t i = 0; i < D; ++i) {
    Aset(A, dofs[row[0]] + offset, dofs[i] + offset, n[i]);
  }
  bset(b, dofs[row[0]], g_value); // set to g instead of 0.0

  // compute the row with tau1
  std::vector<double> a01_tau1(nb_cols_01);
  for (std::size_t i = 0; i < nb_cols_01; ++i) {
    a01_tau1[i] = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      a01_tau1[i] = 0.0;
    }
  }

  for (std::size_t i = 0; i < D; ++i) {
    Aset(A, dofs[row[1]] + offset, dofs[i] + offset, t1[i]);
  }

  A01.setrow(dofs[row[1]] + offset, a01_indices[row[1]], a01_tau1);
  bset(b, dofs[row[1]], 0.0); // before: l_tau1
  if (D == 3) {

    for (std::size_t i = 0; i < D; ++i) {
      Aset(A, dofs[row[2]] + offset, dofs[i] + offset, t2[i]);
    }

    A01.setrow(dofs[row[2]] + offset, a01_indices[row[2]], a01_tau1);
    bset(b, dofs[row[2]], 0.0); // before: l_tau1
  }
}
