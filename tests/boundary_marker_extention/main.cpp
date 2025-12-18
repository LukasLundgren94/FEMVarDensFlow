// Copyright (c) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/io/File.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>

#include "Poisson.h"

using std::size_t;
using std::string;
using namespace dolfin;

namespace {

std::string to_lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool ends_with(const std::string &value, const std::string &suffix) {
  if (suffix.size() > value.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

void read_mesh(dolfin::Mesh &mesh, const std::string &filename) {
  const std::string lower = to_lower(filename);
  if (ends_with(lower, ".xdmf") || ends_with(lower, ".xmf")) {
    dolfin::XDMFFile file(mesh.mpi_comm(), filename);
    file.read(mesh);
  } else {
    dolfin::File file(mesh.mpi_comm(), filename);
    file >> mesh;
  }
}

void write_mesh(const dolfin::Mesh &mesh, const std::string &filename) {
  const std::string lower = to_lower(filename);
  if (ends_with(lower, ".xdmf") || ends_with(lower, ".xmf")) {
    dolfin::XDMFFile file(mesh.mpi_comm(), filename);
    file.write(mesh);
  } else {
    dolfin::File file(mesh.mpi_comm(), filename);
    file << mesh;
  }
}
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int return_code = 0;
  try {
    const int rank = []() {
      int r = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
      return r;
    }();

    if (argc < 3) {
      if (rank == 0) {
        std::cerr << "Usage: " << argv[0] << " <input_mesh> <output_mesh>"
                  << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    dolfin::parameters["std_out_all_processes"] = false;

    {
      auto mesh = std::make_shared<dolfin::Mesh>(MPI_COMM_WORLD);
      read_mesh(*mesh, argv[1]);

      dolfin::info("Loaded mesh with %zu cells and %zu vertices",
                   mesh->num_cells(), mesh->num_vertices());

      std::string const file_name_surface_mesh = argv[2];
      dolfin::XDMFFile mvc_xdmf(mesh->mpi_comm(), file_name_surface_mesh);
      dolfin::MeshValueCollection<std::size_t> mvc(mesh);

      mvc_xdmf.read(
          mvc,
          "name_to_read"); // read Attribute "name" from the right Grid

      dolfin::MeshFunction<std::size_t> cell_markers(mesh, mvc);
      auto sub_domains = std::make_shared<dolfin::MeshFunction<std::size_t>>(
          mesh, mesh->topology().dim() - 1, 0);
      *sub_domains = cell_markers;

      // Build a first-order Lagrange space and interpolate facet markers to vertices.
      auto V = std::make_shared<Poisson::Form_a_FunctionSpace_0>(mesh);
      auto boundary_indicator = std::make_shared<dolfin::Function>(V);

      // Ensure vertex-facet connectivity is available before accumulating values.
      mesh->init(0, mesh->topology().dim() - 1);
      mesh->init(mesh->topology().dim(), mesh->topology().dim() - 1);

      const auto dofmap = V->dofmap();
      const std::size_t num_vertex_dofs = dofmap->num_entity_dofs(0);
      if (num_vertex_dofs == 0)
      {
        throw std::runtime_error(
            "Function space does not have degrees of freedom on vertices.");
      }

      auto cg_vector = boundary_indicator->vector();
      cg_vector->zero();
      const auto cg_range = cg_vector->local_range();
      std::vector<std::size_t> vertex_index(1);
      for (dolfin::VertexIterator v(*mesh); !v.end(); ++v)
      {
        double sum = 0.0;
        std::size_t count = 0;
        for (dolfin::FacetIterator facet(*v); !facet.end(); ++facet)
        {
          const double value = static_cast<double>(cell_markers[facet->index()]);
          if (value > 100.)
          {
            continue;
          }
          sum += value;
          ++count;
        }
        const double value = count > 0 ? std::round(sum / static_cast<double>(count)) : 0.0;
        vertex_index[0] = v->index();
        const auto dofs = dofmap->entity_dofs(*mesh, 0, vertex_index);
        for (std::size_t k = 0; k < dofs.size(); ++k)
        {
          dolfin::la_index dof = dofs[k];
          if (dof >= cg_range.first && dof < cg_range.second)
          {
            double vv = value;
            cg_vector->set(&vv, 1, &dof);
          }
        }
      }

      cg_vector->apply("insert");

      auto V_dg0 = std::make_shared<Poisson::CoefficientSpace_g>(mesh);
      auto boundary_indicator_dg0 = std::make_shared<dolfin::Function>(V_dg0);
      const auto dg0_dofmap = V_dg0->dofmap();
      auto dg0_vector = boundary_indicator_dg0->vector();
      dg0_vector->zero();
      const auto dg0_range = dg0_vector->local_range();
      for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell)
      {
        double sum = 0.0;
        std::size_t count = 0;
        for (dolfin::FacetIterator facet(*cell); !facet.end(); ++facet)
        {
          const double value = static_cast<double>(cell_markers[facet->index()]);
          if (value > 100.)
          {
            continue;
          }
          sum += value;
          ++count;
        }
        const double value = count > 0 ? std::round(sum / static_cast<double>(count)) : 0.0;
        const auto cell_dofs = dg0_dofmap->cell_dofs(cell->index());
        const auto num_cell_dofs = static_cast<std::size_t>(cell_dofs.size());
        for (std::size_t k = 0; k < num_cell_dofs; ++k)
        {
          dolfin::la_index dof = cell_dofs[k];
          if (dof >= dg0_range.first && dof < dg0_range.second)
          {
            double vv = value;
            dg0_vector->set(&vv, 1, &dof);
          }
        }
      }

      dg0_vector->apply("insert");

      dolfin::XDMFFile indicator_file(mesh->mpi_comm(), "boundary_indicator_p1.xdmf");
      indicator_file.write(*boundary_indicator);

      dolfin::XDMFFile indicator_dg0_file(mesh->mpi_comm(), "boundary_indicator_dg0.xdmf");
      indicator_dg0_file.write(*boundary_indicator_dg0);

      Poisson::Form_a poisson_a(V, V);
      Poisson::Form_L poisson_L(V);
      auto g_coefficient = std::static_pointer_cast<const dolfin::GenericFunction>(boundary_indicator_dg0);
      poisson_L.g = g_coefficient;

      dolfin::Matrix A;
      dolfin::Vector b;
      assemble(A, poisson_a);
      assemble(b, poisson_L);

      auto solution = std::make_shared<dolfin::Function>(V);
      dolfin::solve(A, *solution->vector(), b);

      dolfin::XDMFFile solution_file(mesh->mpi_comm(), "poisson_solution.xdmf");
      solution_file.write(*solution);

      dolfin::MeshFunction<double> cell_values(mesh, mesh->topology().dim(), 0.0);
      dolfin::Array<double> value_array(1);
      const std::size_t gdim = mesh->geometry().dim();
      dolfin::Array<double> point_array(gdim);
      for (dolfin::CellIterator cell(*mesh); !cell.end(); ++cell)
      {
        const auto midpoint = cell->midpoint();
        for (std::size_t i = 0; i < gdim; ++i)
        {
          point_array[i] = midpoint[i];
        }
        solution->eval(value_array, point_array);
        if (value_array[0] > 0.9) {
          cell_values[cell->index()] = 1.0;
        }
        else {
          cell_values[cell->index()] = 0.0;
        }
      }

      dolfin::XDMFFile cell_data_file(mesh->mpi_comm(), "poisson_solution_cell_avg.xdmf");
      cell_data_file.write(cell_values);

      dolfin::info("Writing smoothed mesh to %s", argv[3]);
      write_mesh(*mesh, argv[3]);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return_code = 1;
  }
  MPI_Finalize();
  return return_code;
}
