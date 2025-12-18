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

#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/io/File.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/parameter/GlobalParameters.h>

using std::size_t;
using std::string;

namespace
{
std::string to_lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool ends_with(const std::string &value, const std::string &suffix)
{
  if (suffix.size() > value.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

void read_mesh(dolfin::Mesh &mesh, const std::string &filename)
{
  const std::string lower = to_lower(filename);
  if (ends_with(lower, ".xdmf") || ends_with(lower, ".xmf"))
  {
    dolfin::XDMFFile file(mesh.mpi_comm(), filename);
    file.read(mesh);
  }
  else
  {
    dolfin::File file(mesh.mpi_comm(), filename);
    file >> mesh;
  }
}

void smooth_manual(dolfin::Mesh& mesh, std::size_t num_iterations)
{
  dolfin::log(dolfin::PROGRESS, "Smoothing mesh: %s", mesh.str(false).c_str());

  if (mesh.geometry().degree() != 1)
  {
    dolfin::dolfin_error("MeshSmoothing.cpp",
                         "smooth mesh",
                         "This function does not support higher-order mesh geometry");
  }

  // Make sure we have cell-facet connectivity
  mesh.init(mesh.topology().dim(), mesh.topology().dim() - 1);

  // Make sure we have vertex-edge connectivity
  mesh.init(0, 1);

  // Make sure the mesh is ordered
  mesh.order();

  // Mark vertices on the boundary so we may skip them
  dolfin::BoundaryMesh boundary(mesh, "exterior");
  const dolfin::MeshFunction<std::size_t> vertex_map = boundary.entity_map(0);
  auto mesh_ptr = dolfin::reference_to_no_delete_pointer(mesh);
  dolfin::MeshFunction<bool> on_boundary(mesh_ptr, 0);
  on_boundary = false;
  if (boundary.num_vertices() > 0)
  {
    for (dolfin::VertexIterator v(boundary); !v.end(); ++v)
      on_boundary[vertex_map[*v]] = true;
  }

  // Iterate over all vertices
  const std::size_t d = mesh.geometry().dim();
  std::vector<double> xx(d);
  for (std::size_t iteration = 0; iteration < num_iterations; iteration++)
  {
    for (dolfin::VertexIterator v(mesh); !v.end(); ++v)
    {
      // Skip vertices on the boundary
      // if (on_boundary[*v])
        // continue;

      // Get coordinates of vertex
      const dolfin::Point p = v->point();

      // Compute center of mass of neighboring vertices
      for (std::size_t i = 0; i < d; i++) xx[i] = 0.0;
      std::size_t num_neighbors = 0;
      for (dolfin::EdgeIterator e(*v); !e.end(); ++e)
      {
        // Get the other vertex
        dolfin_assert(e->num_entities(0) == 2);
        std::size_t other_index = e->entities(0)[0];
        if (other_index == v->index())
          other_index = e->entities(0)[1];

        // Create the vertex
        dolfin::Vertex vn(mesh, other_index);

        // Skip the vertex itself
        if (v->index() == vn.index())
          continue;
        num_neighbors += 1;

        // Compute center of mass
        const double* xn = vn.x();
        for (std::size_t i = 0; i < d; i++)
          xx[i] += xn[i];
      }
      for (std::size_t i = 0; i < d; i++)
        xx[i] /= static_cast<double>(num_neighbors);

      // Compute closest distance to boundary of star
      double rmin = 0.0;
      for (dolfin::CellIterator c(*v); !c.end(); ++c)
      {
        // Get local number of vertex relative to facet
        const std::size_t local_vertex = c->index(*v);

        // Get normal of corresponding facet
        dolfin::Point n = c->normal(local_vertex);

        // Get first vertex in facet
        dolfin::Facet f(mesh, c->entities(mesh.topology().dim() - 1)[local_vertex]);
        dolfin::VertexIterator fv(f);

        // Compute length of projection of v - fv onto normal
        const double r = std::abs(n.dot(p - fv->point()));
        if (rmin == 0.0)
          rmin = r;
        else
          rmin = std::min(rmin, r);
      }

      // Move vertex at most a distance rmin / 2
      double r = 0.0;
      for (std::size_t i = 0; i < d; i++)
      {
        const double dx = xx[i] - p[i];
        r += dx*dx;
      }
      r = std::sqrt(r);
      if (r < DOLFIN_EPS)
        continue;
      rmin = std::min(0.5*rmin, r);

      std::vector<double> new_vertex(mesh.geometry().x(v->index()),
                                     mesh.geometry().x(v->index()) + d);
      for (std::size_t i = 0; i < d; i++)
        new_vertex[i] += rmin*(xx[i] - p[i])/r;
      mesh.geometry().set(v->index(), new_vertex.data());

    }
  }

  if (num_iterations > 1)
    dolfin::log(dolfin::PROGRESS, "Mesh smoothing repeated %d times.", num_iterations);
}

void write_mesh(const dolfin::Mesh &mesh, const std::string &filename)
{
  const std::string lower = to_lower(filename);
  if (ends_with(lower, ".xdmf") || ends_with(lower, ".xmf"))
  {
    dolfin::XDMFFile file(mesh.mpi_comm(), filename);
    file.write(mesh);
  }
  else
  {
    dolfin::File file(mesh.mpi_comm(), filename);
    file << mesh;
  }
}

std::vector<std::vector<std::size_t>> build_vertex_graph(dolfin::Mesh &mesh)
{
  mesh.init();
  std::vector<std::vector<std::size_t>> adjacency(mesh.num_vertices());
  for (dolfin::CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::vector<std::size_t> cell_vertices;
    cell_vertices.reserve(cell->num_entities(0));
    for (dolfin::VertexIterator v(*cell); !v.end(); ++v)
      cell_vertices.push_back(v->index());
    for (std::size_t i = 0; i < cell_vertices.size(); ++i)
      for (std::size_t j = 0; j < cell_vertices.size(); ++j)
        if (i != j)
          adjacency[cell_vertices[i]].push_back(cell_vertices[j]);
  }
  for (auto &neighbors : adjacency)
  {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  return adjacency;
}

std::vector<bool> mark_boundary_vertices(dolfin::Mesh &mesh)
{
  std::vector<bool> boundary(mesh.num_vertices(), false);
  if (mesh.num_vertices() == 0)
    return boundary;

  dolfin::BoundaryMesh surface(mesh, "exterior");
  auto vertex_map = surface.entity_map(0);
  for (dolfin::VertexIterator v(surface); !v.end(); ++v)
  {
    const std::size_t parent = vertex_map[*v];
    if (parent < boundary.size())
      boundary[parent] = true;
  }
  return boundary;
}

void laplacian_smooth(dolfin::Mesh &mesh,
                      std::size_t iterations,
                      double relaxation,
                      bool include_boundary)
{
  if (iterations == 0 || relaxation <= 0.0)
    return;

  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t nverts = mesh.num_vertices();

  auto adjacency = build_vertex_graph(mesh);
  auto is_boundary = mark_boundary_vertices(mesh);

  std::vector<double> updated(nverts * gdim, 0.0);
  std::vector<double> buffer(gdim, 0.0);
  auto &geometry = mesh.geometry();

  for (std::size_t step = 0; step < iterations; ++step)
  {
    for (std::size_t v = 0; v < nverts; ++v)
    {
      if ((!include_boundary && is_boundary[v]) || adjacency[v].empty())
      {
        for (std::size_t d = 0; d < gdim; ++d)
          updated[v * gdim + d] = geometry.x(v, d);
        continue;
      }

      std::fill(buffer.begin(), buffer.end(), 0.0);
      for (const auto neighbor : adjacency[v])
        for (std::size_t d = 0; d < gdim; ++d)
          buffer[d] += geometry.x(neighbor, d);

      const double scale = 1.0 / static_cast<double>(adjacency[v].size());
      for (std::size_t d = 0; d < gdim; ++d)
      {
        const double averaged = buffer[d] * scale;
        const double current = geometry.x(v, d);
        updated[v * gdim + d] = (1.0 - relaxation) * current + relaxation * averaged;
      }
    }

    for (std::size_t v = 0; v < nverts; ++v)
    {
      for (std::size_t d = 0; d < gdim; ++d)
        buffer[d] = updated[v * gdim + d];
      geometry.set(v, buffer.data());
    }
  }
}

void apply_graph_operator(const std::vector<std::vector<std::size_t>> &adjacency,
                          const std::vector<double> &diag,
                          const std::vector<double> &x,
                          std::vector<double> &out)
{
  const std::size_t n = adjacency.size();
  out.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
  {
    double value = diag[i] * x[i];
    for (const auto neighbour : adjacency[i])
      value -= x[neighbour];
    out[i] = value;
  }
}

double dot_product(const std::vector<double> &a,
                   const std::vector<double> &b)
{
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

bool conjugate_gradient_graph(const std::vector<std::vector<std::size_t>> &adjacency,
                              const std::vector<double> &diag,
                              const std::vector<double> &b,
                              std::vector<double> &x,
                              std::size_t max_iterations,
                              double tolerance)
{
  const std::size_t n = adjacency.size();
  if (n == 0)
    return true;

  std::vector<double> r(n, 0.0);
  std::vector<double> p(n, 0.0);
  std::vector<double> Ap(n, 0.0);

  apply_graph_operator(adjacency, diag, x, Ap);
  for (std::size_t i = 0; i < n; ++i)
    r[i] = b[i] - Ap[i];
  p = r;

  double rs_old = dot_product(r, r);
  const double tol2 = tolerance * tolerance;
  if (rs_old < tol2)
    return true;

  for (std::size_t iter = 0; iter < max_iterations; ++iter)
  {
    apply_graph_operator(adjacency, diag, p, Ap);
    double denom = dot_product(p, Ap);
    if (std::abs(denom) < std::numeric_limits<double>::epsilon())
      break;

    const double alpha = rs_old / denom;
    for (std::size_t i = 0; i < n; ++i)
    {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = dot_product(r, r);
    if (rs_new < tol2)
      return true;

    const double beta = rs_new / rs_old;
    for (std::size_t i = 0; i < n; ++i)
      p[i] = r[i] + beta * p[i];
    rs_old = rs_new;
  }
  return false;
}

void poisson_smooth(dolfin::Mesh &mesh,
                    double lambda,
                    std::size_t max_iterations,
                    double tolerance,
                    double top_fix_eps)
{
  if (lambda <= 0.0)
    throw std::invalid_argument("lambda for poisson smoothing must be positive");

  const auto adjacency = build_vertex_graph(mesh);
  const std::size_t nverts = adjacency.size();
  if (nverts == 0)
    return;

  auto &geometry = mesh.geometry();
  const std::size_t gdim = geometry.dim();

  std::vector<double> solution_storage(nverts * gdim, 0.0);
  std::vector<double> original_coords(nverts * gdim, 0.0);
  for (std::size_t v = 0; v < nverts; ++v)
    for (std::size_t d = 0; d < gdim; ++d)
      original_coords[v * gdim + d] = geometry.x(v, d);

  std::vector<bool> fixed_mask(nverts, false);
  if (top_fix_eps >= 0.0)
  {
    if (gdim < 2)
    {
      dolfin::warning("Top boundary fixation requested but mesh is %zu-dimensional. Ignoring.", gdim);
    }
    else
    {
      double ymax = -std::numeric_limits<double>::infinity();
      for (std::size_t v = 0; v < nverts; ++v)
        ymax = std::max(ymax, original_coords[v * gdim + 1]);
      for (std::size_t v = 0; v < nverts; ++v)
      {
        const double y = original_coords[v * gdim + 1];
        if (std::abs(y - ymax) < top_fix_eps)
          fixed_mask[v] = true;
      }
    }
  }

  std::vector<int> free_index(nverts, -1);
  std::vector<std::size_t> free_vertices;
  free_vertices.reserve(nverts);
  for (std::size_t v = 0; v < nverts; ++v)
  {
    if (fixed_mask[v])
      continue;
    free_index[v] = static_cast<int>(free_vertices.size());
    free_vertices.push_back(v);
  }

  const std::size_t n_free = free_vertices.size();
  if (n_free == 0)
  {
    dolfin::info("All vertices are fixed; skipping Poisson smoothing.");
    return;
  }

  std::vector<std::vector<std::size_t>> free_adjacency(n_free);
  std::vector<std::vector<std::size_t>> fixed_neighbors(n_free);
  std::vector<double> diag(n_free, 0.0);

  for (std::size_t idx = 0; idx < n_free; ++idx)
  {
    const std::size_t vertex = free_vertices[idx];
    const auto &nbrs = adjacency[vertex];
    diag[idx] = lambda + static_cast<double>(nbrs.size());
    for (const auto nbr : nbrs)
    {
      if (fixed_mask[nbr])
        fixed_neighbors[idx].push_back(nbr);
      else
      {
        const int nbr_index = free_index[nbr];
        if (nbr_index >= 0)
          free_adjacency[idx].push_back(static_cast<std::size_t>(nbr_index));
        else
          dolfin::error("Invalid neighbour index encountered while building Poisson system.");
      }
    }
  }

  std::vector<double> b(n_free, 0.0);
  std::vector<double> x(n_free, 0.0);

  for (std::size_t d = 0; d < gdim; ++d)
  {
    for (std::size_t idx = 0; idx < n_free; ++idx)
    {
      const std::size_t v = free_vertices[idx];
      const double value = original_coords[v * gdim + d];
      double rhs = lambda * value;
      for (const auto neighbour : fixed_neighbors[idx])
        rhs += original_coords[neighbour * gdim + d];
      b[idx] = rhs;
      x[idx] = value;
    }

    const bool converged =
        conjugate_gradient_graph(free_adjacency, diag, b, x, max_iterations, tolerance);
    if (!converged)
      dolfin::warning("Vector Poisson smoothing did not converge for component %zu", d);

    for (std::size_t idx = 0; idx < n_free; ++idx)
    {
      const std::size_t v = free_vertices[idx];
      solution_storage[v * gdim + d] = x[idx];
    }
    for (std::size_t v = 0; v < nverts; ++v)
    {
      if (fixed_mask[v])
        solution_storage[v * gdim + d] = original_coords[v * gdim + d];
    }
  }

  std::vector<double> buffer(gdim, 0.0);
  for (std::size_t v = 0; v < nverts; ++v)
  {
    for (std::size_t d = 0; d < gdim; ++d)
      buffer[d] = solution_storage[v * gdim + d];
    geometry.set(v, buffer.data());
  }
}
} // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int return_code = 0;
  try
  {
    const int rank = []() {
      int r = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
      return r;
    }();

    if (argc < 3)
    {
      if (rank == 0)
      {
        std::cerr << "Usage: " << argv[0]
                  << " <input_mesh> <output_mesh> [iterations] [relaxation] [method] [method arguments...]" << std::endl;
        std::cerr << "  methods: manual (default), laplace, laplace-all, poisson" << std::endl;
        std::cerr << "    poisson arguments (optional): lambda [max_iter tol fix_eps]" << std::endl;
        std::cerr << "      fix_eps >= 0 fixes vertices with |y - y_max| < fix_eps during smoothing" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }

    const std::size_t iterations = (argc > 3) ? std::stoul(argv[3]) : 5u;
    const double relaxation = (argc > 4) ? std::stod(argv[4]) : 0.5;
    const std::string method = (argc > 5) ? to_lower(std::string(argv[5])) : "manual";

    dolfin::parameters["std_out_all_processes"] = false;

    {
      dolfin::Mesh mesh(MPI_COMM_WORLD);
      read_mesh(mesh, argv[1]);

      dolfin::info("Loaded mesh with %zu cells and %zu vertices", mesh.num_cells(),
                   mesh.num_vertices());

      if (method == "manual")
      {
        dolfin::info("Applying manual Laplacian-style smoothing (%zu iterations)", iterations);
        smooth_manual(mesh, iterations);
      }
      else if (method == "laplace")
      {
        dolfin::info("Applying Laplacian smoothing to interior vertices (%zu iterations, relaxation=%g)",
                     iterations, relaxation);
        laplacian_smooth(mesh, iterations, relaxation, false);
      }
      else if (method == "laplace-all")
      {
        dolfin::info("Applying Laplacian smoothing to all vertices (%zu iterations, relaxation=%g)",
                     iterations, relaxation);
        laplacian_smooth(mesh, iterations, relaxation, true);
      }
      else if (method == "poisson")
      {
        const double lambda = (argc > 6) ? std::stod(argv[6]) : 1.0;
        const std::size_t max_cg = (argc > 7) ? std::stoul(argv[7]) : 200u;
        const double tol = (argc > 8) ? std::stod(argv[8]) : 1e-10;
        const double fix_eps = (argc > 9) ? std::stod(argv[9]) : -1.0;
        if (fix_eps >= 0.0)
        {
          dolfin::info("Applying vector Poisson smoothing (lambda=%g, max_iter=%zu, tol=%g) with fixed top eps=%g",
                       lambda, max_cg, tol, fix_eps);
        }
        else
        {
          dolfin::info("Applying vector Poisson smoothing (lambda=%g, max_iter=%zu, tol=%g)",
                       lambda, max_cg, tol);
        }
        poisson_smooth(mesh, lambda, max_cg, tol, fix_eps);
      }
      else
      {
        dolfin::warning("Unknown smoothing method '%s'. Falling back to manual smoothing.",
                        method.c_str());
        smooth_manual(mesh, iterations);
      }

      dolfin::info("Writing smoothed mesh to %s", argv[2]);
      write_mesh(mesh, argv[2]);
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return_code = 1;
  }
  MPI_Finalize();
  return return_code;
}
