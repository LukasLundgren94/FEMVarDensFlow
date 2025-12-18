// Copyright (C) 2007-2017 Murtazo Nazarov
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson, 2008-2010. 
// Modifed by Christian Helanow, 2023.
// Modified by Lukas Lundgren 2025.
// 
// First added:  2007-05-01
// Last changed: 2017-01-01

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include <dolfin/common/MPI.h>

#include <dolfin/la/Vector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/DofMap.h>

#include "VertexNormal.h"

using namespace fem_varden_flow;

//-----------------------------------------------------------------------------
VertexNormal::VertexNormal(std::shared_ptr<const Mesh> mesh) :
    mesh_(mesh)
{
  const std::size_t D = mesh_->topology().dim();

  Fnormal = new MeshFunction<double> [D];
  Ftau1 = new MeshFunction<double> [D];
  Ftau2 = new MeshFunction<double> [D];

  for (std::size_t i = 0; i < D; i++)
  {
    Fnormal[i].init(mesh_, 0);
    Ftau1[i].init(mesh_, 0);
    Ftau2[i].init(mesh_, 0);
    Fnormal[i] = 0.0;
    Ftau1[i] = 0.0;
    Ftau2[i] = 0.0;
  }

  boundary_ = std::make_shared<BoundaryMesh>(*mesh_, "exterior");

  vertex_map_ = boundary_->entity_map(0);
  cell_map_ = boundary_->entity_map(boundary_->topology().dim());

  parameters = default_parameters();

  compute_normal();
}

//-----------------------------------------------------------------------------
VertexNormal::~VertexNormal()
{
  clear();
}

//-----------------------------------------------------------------------------
void VertexNormal::clear()
{
  if (Fnormal) delete[] Fnormal;
  if (Ftau1) delete[] Ftau1;
  if (Ftau2) delete[] Ftau2;
}

//-----------------------------------------------------------------------------
void VertexNormal::compute_normal()
{
  const std::size_t D = mesh_->topology().dim();

  if (boundary_->num_vertices() > 0)
  {
    set_log_active(true);
    mesh_->init();

    std::size_t rank = MPI::rank(mesh_->mpi_comm());

    // get the total number of sharing vertices
    std::size_t N = 0;
    for (VertexIterator v(*boundary_, "all"); !v.end(); ++v)
    {
      if ((*v).is_shared())
      {
        N += (*v).sharing_processes().size();
      }
    }

    MPI_Status mpi_stat[N];
    MPI_Request mpi_req[N];

    std::vector<std::size_t> shared_v_glob(N);
    std::vector<double> shared_weight(N);
    std::vector<std::vector<double>> shared_n_block(N, std::vector<double>(D));
//    std::vector<std::size_t> v_glob = mesh_->topology().global_indices(0);
    const auto& v_glob = mesh_->topology().global_indices(0);

    std::size_t ind = 0;
    for (VertexIterator v(*boundary_, "all"); !v.end(); ++v)
    {
      if ((*v).is_shared())
      {
        std::size_t loc_id = vertex_map_[*v];

        // Send-receive between sharing processes
        const std::set<unsigned int>& sharing_processes =
            (*v).sharing_processes();
        for (std::set<unsigned int>::iterator it = sharing_processes.begin();
            it != sharing_processes.end(); ++it, ++ind)
        {
          std::vector<double> n(D, 0.0);
          double weight = 0.0;
          std::size_t v_gl_id = v_glob[loc_id];

          // Collect all facet normals connected to the vertex
          get_facet_normal(*v, n, weight);

          // Send-receive the data
          MPI_Isend(&v_gl_id, 1, MPI_UNSIGNED_LONG, *it, 11, mesh_->mpi_comm(),
                    &mpi_req[ind]);
          MPI_Irecv(&shared_v_glob[ind], 1, MPI_UNSIGNED_LONG, *it, 11,
                    mesh_->mpi_comm(), &mpi_req[ind]);

          MPI_Isend(&weight, 1, MPI_DOUBLE, *it, 22, mesh_->mpi_comm(),
                    &mpi_req[ind]);
          MPI_Irecv(&shared_weight[ind], 1, MPI_DOUBLE, *it, 22,
                    mesh_->mpi_comm(), &mpi_req[ind]);

          MPI_Isend(&n[0], D, MPI_DOUBLE, *it, 33, mesh_->mpi_comm(),
                    &mpi_req[ind]);
          MPI_Irecv(&shared_n_block[ind][0], D, MPI_DOUBLE, *it, 33,
                    mesh_->mpi_comm(), &mpi_req[ind]);
        }
      }
    }
    MPI_Waitall(N, mpi_req, mpi_stat);

    for (VertexIterator v(*boundary_, "all"); !v.end(); ++v)
    {
      std::vector<double> n(D);
      double weight = 0.0;

      // Collect all facet normals connected to the vertex
      get_facet_normal(*v, n, weight);

      std::size_t loc_id = vertex_map_[*v];

      if ((*v).is_shared())
      {
        const std::set<unsigned int>& sharing_processes =
            (*v).sharing_processes();
        for (std::size_t i = 0; i < shared_v_glob.size(); ++i)
        {
          std::size_t v_gl_id = v_glob[loc_id];
          if (v_gl_id == shared_v_glob[i])
          {
            weight += shared_weight[i];
            for (std::size_t j = 0; j < D; ++j)
            {
              n[j] += shared_n_block[i][j];
            }
          }
        }
      }

      std::vector<double> vertex_normal(D);
      std::vector<double> vertex_tau(D);
      std::vector<double> vertex_tau1(D);
      std::vector<double> vertex_tau2(D);
      double magnitude = 0.0;

      for (std::size_t j = 0; j < D; ++j)
      {
        vertex_normal[j] = n[j] / weight;
        magnitude += vertex_normal[j]*vertex_normal[j];
      }

      // normalize the normal vector
      for (std::size_t j = 0; j < D; ++j)
			{
				vertex_normal[j] /= std::sqrt(magnitude);
			}

      if (D == 2)
      {
        vertex_tau[0] = vertex_normal[1];
        vertex_tau[1] = -vertex_normal[0];
        for (std::size_t i = 0; i < D; ++i)
        {
          Fnormal[i].set_value(loc_id, vertex_normal[i]);
          Ftau1[i].set_value(loc_id, vertex_tau[i]);
        }
      }
      else if (D == 3)
      {
        compute_tangent(vertex_normal, vertex_tau1, vertex_tau2);
        for (std::size_t i = 0; i < D; ++i)
        {
          Fnormal[i].set_value(loc_id, vertex_normal[i]);
          Ftau1[i].set_value(loc_id, vertex_tau1[i]);
          Ftau2[i].set_value(loc_id, vertex_tau2[i]);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
void VertexNormal::get_facet_normal(Vertex const& v,
                                    std::vector<double>& n_weighted,
                                    double& sum_weights)
{

  const std::size_t D = mesh_->topology().dim();
  sum_weights = 0.0;
  std::fill(n_weighted.begin(), n_weighted.end(), 0);

  for (CellIterator boundary_cell(v); !boundary_cell.end(); ++boundary_cell)
  {
    // Create mesh facet corresponding to boundary cell
    Facet c_facet(*mesh_, cell_map_[*boundary_cell]);

    // Get cell to which facet belongs (pick first, there is only one)
    Cell f_cell(*mesh_, c_facet.entities(mesh_->topology().dim())[0]);

    // Get local index of facet with respect to the cell
    std::size_t local_facet = f_cell.index(c_facet);

    // Compute the lenght/area of the cell
    double area = f_cell.facet_area(local_facet);
    sum_weights += area;

    // Compute a normal to the boundary
    for (std::size_t i = 0; i < D; ++i)
    {
      double n_i = f_cell.normal(local_facet, i);
      n_weighted[i] += n_i * area;
    }
  }
}

//-----------------------------------------------------------------------------
void VertexNormal::compute_tangent(std::vector<double> const & n,
                                   std::vector<double> & t1,
                                   std::vector<double> & t2)
{
  const std::size_t D = mesh_->topology().dim();

  double t11, t12, t13;
  double t21, t22, t23;
  if (std::fabs(n[0]) >= 0.5 || std::fabs(n[1]) >= 0.5)
  {
    double nn = std::sqrt(n[0] * n[0] + n[1] * n[1]);
    t11 = n[1] / nn;
    t12 = -n[0] / nn;
    t13 = 0.0;
    t21 = -t12 * n[2];
    t22 = t11 * n[2];
    t23 = t12 * n[0] - t11 * n[1];
  }
  else
  {
    double nn = std::sqrt(n[1] * n[1] + n[2] * n[2]);
    t11 = 0.0;
    t12 = -n[2] / nn;
    t13 = n[1] / nn;
    t21 = -t13 * n[1] - t12 * n[2];
    t22 = -t13 * n[0];
    t23 = t12 * n[0];
  }
  t1[0] = t11;
  t1[1] = t12;
  t1[2] = t13;

  t2[0] = t21;
  t2[1] = t22;
  t2[2] = t23;
}

//-----------------------------------------------------------------------------
VertexNormal::SharedData::SharedData()
{
// Do nothing
}
