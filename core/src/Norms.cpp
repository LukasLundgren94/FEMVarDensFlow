// Copyright (C) 2011 Murtazo Nazarov
// Licensed under the MIT License – see the LICENSE file for details
//
// First added:  2011-05-01
// Last changed: 2019-12-22

#include <fstream>
#include <iostream>
#include <memory>

///
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/timing.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/io/File.h>
#include <dolfin/log/Logger.h>
#include <dolfin/log/Progress.h>
#include <dolfin/mesh/Mesh.h>

///
namespace dolfin {
class Function;
class GenericFunction;
class Mesh;
class assemble;
class Form;
class Logger;
class Progress;
class File;
} // namespace dolfin

using dolfin::error;
using dolfin::File;
using dolfin::Form;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::GenericFunction;
using dolfin::info;
using dolfin::Logger;
using dolfin::Mesh;
using dolfin::NoDeleter;
using dolfin::Progress;
using dolfin::set_log_active;
using dolfin::Variable;
using dolfin::warning;

#include "Norms.h"
/// 1D

/// 2D
#include "ufl/norm/L1_Norm_2D_P1.h"
#include "ufl/norm/L1_Norm_V_2D_P1.h"
#include "ufl/norm/L2_Norm_2D_P1.h"
#include "ufl/norm/L2_Norm_V_2D_P1.h"

#include "ufl/norm/L1_Norm_2D_P2.h"
#include "ufl/norm/L1_Norm_V_2D_P2.h"
#include "ufl/norm/L2_Norm_2D_P2.h"
#include "ufl/norm/L2_Norm_V_2D_P2.h"

#include "ufl/norm/L1_Norm_2D_P3.h"
#include "ufl/norm/L1_Norm_V_2D_P3.h"
#include "ufl/norm/L2_Norm_2D_P3.h"
#include "ufl/norm/L2_Norm_V_2D_P3.h"

#include "ufl/norm/L1_Norm_2D_P4.h"
#include "ufl/norm/L1_Norm_V_2D_P4.h"
#include "ufl/norm/L2_Norm_2D_P4.h"
#include "ufl/norm/L2_Norm_V_2D_P4.h"

/// 3D
#include "ufl/norm/L1_Norm_3D_P1.h"
#include "ufl/norm/L2_Norm_3D_P1.h"

#include "ufl/norm/L1_Norm_3D_P2.h"
#include "ufl/norm/L2_Norm_3D_P2.h"

#include "ufl/norm/L1_Norm_3D_P2.h"
#include "ufl/norm/L1_Norm_V_3D_P2.h"
#include "ufl/norm/L2_Norm_3D_P2.h"
#include "ufl/norm/L2_Norm_V_3D_P2.h"

#include "ufl/norm/L1_Norm_3D_P3.h"
#include "ufl/norm/L1_Norm_V_3D_P3.h"
#include "ufl/norm/L2_Norm_3D_P3.h"
#include "ufl/norm/L2_Norm_V_3D_P3.h"

using namespace fem_varden_flow;

// -----------------------------------------------------------------------------
void Norms::get_forms(std::shared_ptr<const Mesh> mesh_,
                      std::shared_ptr<Form> &L1, std::shared_ptr<Form> &L2,
                      std::shared_ptr<FunctionSpace> &V_ex,
                      std::size_t const pol_degree,
                      std::size_t const value_size) {
  const std::size_t D = mesh_->geometry().dim();

  if (D == 1 and value_size == 1) {
    switch (pol_degree) {
    case 1:
      // L1 = std::make_shared < L1_Norm_1D_P1::Functional > (mesh_);
      // L2 = std::make_shared < L2_Norm_1D_P1::Functional > (mesh_);
      // V_ex_ = std::make_shared < L2_Norm_1D_P1::CoefficientSpace_u_ex >
      // (mesh_);
      break;
    default:
      error("Error norms are not computed to P = %d", pol_degree);
    }
  }
  if (D == 2 and value_size == 1) {
    switch (pol_degree) {
    case 1:
      L1 = std::make_shared<L1_Norm_2D_P1::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_2D_P1::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_2D_P1::CoefficientSpace_u_ex>(mesh_);
      break;
    case 2:
      L1 = std::make_shared<L1_Norm_2D_P2::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_2D_P2::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_2D_P2::CoefficientSpace_u_ex>(mesh_);
      break;
    case 3:
      L1 = std::make_shared<L1_Norm_2D_P3::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_2D_P3::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_2D_P3::CoefficientSpace_u_ex>(mesh_);
      break;
    case 4:
      L1 = std::make_shared<L1_Norm_2D_P4::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_2D_P4::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_2D_P4::CoefficientSpace_u_ex>(mesh_);
      break;
    default:
      error("Error norms are not computed to P = %d", pol_degree);
    }
  } else if (D == 2 and value_size == 2) {
    switch (pol_degree) {
    case 1:
      L1 = std::make_shared<L1_Norm_V_2D_P1::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_2D_P1::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_2D_P1::CoefficientSpace_u_ex>(mesh_);
      break;
    case 2:
      L1 = std::make_shared<L1_Norm_V_2D_P2::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_2D_P2::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_2D_P2::CoefficientSpace_u_ex>(mesh_);
      break;
    case 3:
      L1 = std::make_shared<L1_Norm_V_2D_P3::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_2D_P3::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_2D_P3::CoefficientSpace_u_ex>(mesh_);
      break;
    case 4:
      L1 = std::make_shared<L1_Norm_V_2D_P4::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_2D_P4::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_2D_P4::CoefficientSpace_u_ex>(mesh_);
      break;
    default:
      error("Error norms are not computed to P = %d", pol_degree);
    }
  }

  if (D == 3 and value_size == 1) {
    switch (pol_degree) {
    case 1:
      L1 = std::make_shared<L1_Norm_3D_P1::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_3D_P1::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_3D_P1::CoefficientSpace_u_ex>(mesh_);
      break;
    case 2:
      L1 = std::make_shared<L1_Norm_3D_P2::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_3D_P2::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_3D_P2::CoefficientSpace_u_ex>(mesh_);
      break;
    case 3:
      L1 = std::make_shared<L1_Norm_3D_P3::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_3D_P3::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_3D_P3::CoefficientSpace_u_ex>(mesh_);
      break;
    default:
      error("Error norms are not computed to P = %d", pol_degree);
    }
  }

  if (D == 3 and value_size == 3) {
    switch (pol_degree) {
    case 2:
      L1 = std::make_shared<L1_Norm_V_3D_P2::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_3D_P2::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_3D_P2::CoefficientSpace_u_ex>(mesh_);
      break;
    case 3:
      L1 = std::make_shared<L1_Norm_V_3D_P3::Functional>(mesh_);
      L2 = std::make_shared<L2_Norm_V_3D_P3::Functional>(mesh_);
      V_ex = std::make_shared<L2_Norm_V_3D_P3::CoefficientSpace_u_ex>(mesh_);
      break;
    default:
      error("Error norms are not computed to P = %d", pol_degree);
    }
  }
}

// -----------------------------------------------------------------------------
void Norms::compute_error(Function const &Fu, GenericFunction const &Fu_ex,
                          std::string const &file_name) {
  auto mesh_ = Fu.function_space()->mesh();
  std::size_t const pol_degree =
      Fu.function_space()->element()->ufc_element()->degree();
  std::size_t const value_size = Fu.value_size();

  std::shared_ptr<Form> L1;
  std::shared_ptr<Form> L2;
  std::shared_ptr<FunctionSpace> V_ex;

  get_forms(mesh_, L1, L2, V_ex, pol_degree, value_size);

  // compute the norm of solution, needed for relative error
  auto Fzero = std::make_shared<Function>(Fu);
  Fzero->vector()->zero();
  L1->set_coefficient(
      "u_ex", std::shared_ptr<const GenericFunction>(&Fu_ex, NoDeleter()));
  L2->set_coefficient(
      "u_ex", std::shared_ptr<const GenericFunction>(&Fu_ex, NoDeleter()));
  L1->set_coefficient("u", Fzero);
  L2->set_coefficient("u", Fzero);

  //
  double const l1norm_u = assemble(*L1);
  double const l2norm_u = sqrt(assemble(*L2));
  double const linfnorm_u = Fu.vector()->norm("linf");

  /// compute error norms
  // set the coefficients
  L1->set_coefficient("u", std::shared_ptr<const Function>(&Fu, NoDeleter()));
  L2->set_coefficient("u", std::shared_ptr<const Function>(&Fu, NoDeleter()));

  // compute the L1 and L2 norms
  double const l1errnorm = assemble(*L1) / l1norm_u;
  double const l2errnorm = sqrt(assemble(*L2)) / l2norm_u;

  // compute the Linf norm
  Function u_diff(V_ex);
  Function u_ex(V_ex);
  u_ex.interpolate(Fu_ex);
  u_diff.interpolate(Fu);
  u_diff.vector()->axpy(-1.0, *u_ex.vector());
  double const linf_errnorm = u_diff.vector()->norm("linf") / linfnorm_u;

  if (dolfin::MPI::rank(mesh_->mpi_comm()) == 0) {
    // save to file
    std::ostringstream error_fname;
    error_fname << file_name.c_str() << "_P" << pol_degree << ".m";
    std::ofstream errorFile;
    errorFile.open((error_fname.str()).c_str(), std::ios::app);

    errorFile << pol_degree << "\t" << l1errnorm << "\t" << l2errnorm << "\t"
              << linf_errnorm << "\t"
              << Fu.function_space()->mesh()->num_vertices() << "\t"
              << Fu.function_space()->mesh()->num_cells() << "\t"
              << Fu.function_space()->dofmap()->global_dimension() << "\n";
    errorFile.flush();
    errorFile.close();

    /// print the result
    set_log_active(true);
    info("Error norms:");
    std::cout << pol_degree << ",  " << l1errnorm << ",  " << l2errnorm << ",  "
              << linf_errnorm << ",  "
              << Fu.function_space()->mesh()->num_vertices() << ",  "
              << Fu.function_space()->dofmap()->global_dimension() << std::endl;
  }

}

// -----------------------------------------------------------------------------
void Norms::compute_absolute_error(Function const &Fu, GenericFunction const &Fu_ex,
                          std::string const &file_name) {
  auto mesh_ = Fu.function_space()->mesh();
  std::size_t const pol_degree =
      Fu.function_space()->element()->ufc_element()->degree();
  std::size_t const value_size = Fu.value_size();

  std::shared_ptr<Form> L1;
  std::shared_ptr<Form> L2;
  std::shared_ptr<FunctionSpace> V_ex;

  get_forms(mesh_, L1, L2, V_ex, pol_degree, value_size);

  // compute the norm of solution, needed for relative error
  auto Fzero = std::make_shared<Function>(Fu);
  Fzero->vector()->zero();
  L1->set_coefficient(
      "u_ex", std::shared_ptr<const GenericFunction>(&Fu_ex, NoDeleter()));
  L2->set_coefficient(
      "u_ex", std::shared_ptr<const GenericFunction>(&Fu_ex, NoDeleter()));
  L1->set_coefficient("u", Fzero);
  L2->set_coefficient("u", Fzero);

  //
  double const l1norm_u = assemble(*L1);
  double const l2norm_u = sqrt(assemble(*L2));
  double const linfnorm_u = Fu.vector()->norm("linf");

  /// compute error norms
  // set the coefficients
  L1->set_coefficient("u", std::shared_ptr<const Function>(&Fu, NoDeleter()));
  L2->set_coefficient("u", std::shared_ptr<const Function>(&Fu, NoDeleter()));

  // compute the L1 and L2 norms
  double const l1errnorm = assemble(*L1);
  double const l2errnorm = sqrt(assemble(*L2));

  // compute the Linf norm
  Function u_diff(V_ex);
  Function u_ex(V_ex);
  u_ex.interpolate(Fu_ex);
  u_diff.interpolate(Fu);
  u_diff.vector()->axpy(-1.0, *u_ex.vector());
  double const linf_errnorm = u_diff.vector()->norm("linf");

  if (dolfin::MPI::rank(mesh_->mpi_comm()) == 0) {
    // save to file
    std::ostringstream error_fname;
    error_fname << file_name.c_str() << "_P" << pol_degree << ".m";
    std::ofstream errorFile;
    errorFile.open((error_fname.str()).c_str(), std::ios::app);

    errorFile << pol_degree << "\t" << l1errnorm << "\t" << l2errnorm << "\t"
              << linf_errnorm << "\t"
              << Fu.function_space()->mesh()->num_vertices() << "\t"
              << Fu.function_space()->mesh()->num_cells() << "\t"
              << Fu.function_space()->dofmap()->global_dimension() << "\n";
    errorFile.flush();
    errorFile.close();

    /// print the result
    set_log_active(true);
    info("Error norms:");
    std::cout << pol_degree << ",  " << l1errnorm << ",  " << l2errnorm << ",  "
              << linf_errnorm << ",  "
              << Fu.function_space()->mesh()->num_vertices() << ",  "
              << Fu.function_space()->dofmap()->global_dimension() << std::endl;
  }

}