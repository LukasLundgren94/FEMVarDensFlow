// Copyright (C) 2016 Murtazo Nazarov
// Copyright (C) 2023-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
//
//
//
// First added:  2016-02-03
// Last changed: 2019-12-22

#include <dolfin/common/timing.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/parameter/Parameter.h>

///
namespace dolfin {
class Function;
class GenericFunction;
class Mesh;
class assemble;
class Form;
class Logger;
class Progress;
} // namespace dolfin

using dolfin::Constant;
using dolfin::DirichletBC;
using dolfin::error;
using dolfin::Form;
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::GenericFunction;
using dolfin::info;
using dolfin::Logger;
using dolfin::Mesh;
using dolfin::Progress;
using dolfin::set_log_active;
using dolfin::Variable;
using dolfin::warning;

#include "PDESmooth.h"

using namespace fem_varden_flow;

// -----------------------------------------------------------------------------
PDESmooth::PDESmooth(std::shared_ptr<const FunctionSpace> V,
                     std::shared_ptr<const Function> Fh, Parameters &p)
    : V_(V), Fh_(Fh) {
  parameters = p;
  init();
}

// -----------------------------------------------------------------------------
PDESmooth::PDESmooth(std::shared_ptr<const FunctionSpace> V,
                     std::shared_ptr<const Function> Fh, Parameters &p,
                     std::shared_ptr<std::vector<DirichletBC *>> bcs)
    : V_(V), Fh_(Fh) {
  parameters = p;
  bcs_ = bcs;
  init();
}

// -----------------------------------------------------------------------------
PDESmooth::~PDESmooth() {}

// -----------------------------------------------------------------------------
void PDESmooth::init() {
  const Mesh &mesh_ = *V_->mesh();
  const std::size_t D = mesh_.topology().dim();
  int const pol_degree = parameters["PolDegree"];
  double const smoothing_weight = parameters["smoothing_weight"];

  auto Fweight = std::make_shared<Constant>(smoothing_weight);

  // Fu_ = std::make_shared < Function > (V_);

  std::string signature = V_->element()->signature().c_str();
  std::string signature_Fh =
      Fh_->function_space()->element()->signature().c_str();

  set_log_active(true);
  info("Space is %s, %s", signature.c_str(), signature_Fh.c_str());

  if (signature_Fh == Q_CG1_2D or signature_Fh == Q_CG2_2D or
      signature_Fh == Q_CG3_2D or signature_Fh == Q_CG4_2D) {
    // smoothing using CG 1 mesh-size
    if (signature == Q_CG1_2D) {
      a_ = std::make_shared<PDESmoothCGh_2D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_2D_P1::LinearForm>(V_);
    } else if (signature == Q_CG2_2D) {
      a_ = std::make_shared<PDESmoothCGh_2D_P2::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_2D_P2::LinearForm>(V_);
    } else if (signature == Q_CG3_2D) {
      a_ = std::make_shared<PDESmoothCGh_2D_P3::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_2D_P3::LinearForm>(V_);
    } else if (signature == V_CG1_2D) {
      a_ = std::make_shared<PDESmoothVCG1h_2D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCG1h_2D_P1::LinearForm>(V_);
    } else if (signature == V_CG2_2D) {
      a_ = std::make_shared<PDESmoothVCG1h_2D_P2::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCG1h_2D_P2::LinearForm>(V_);
    } else if (signature == V_CG3_2D) {
      a_ = std::make_shared<PDESmoothVCG1h_2D_P3::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCG1h_2D_P3::LinearForm>(V_);
    } else
      error("Function spaces for P%d does not exist", pol_degree);
  } else if (signature_Fh == Q_DG0_2D) {
    // smoothing using DG mesh-size
    if (signature == Q_CG1_2D) {
      a_ = std::make_shared<PDESmooth_2D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmooth_2D_P1::LinearForm>(V_);
    }
  } else if (signature_Fh == V_CG1_2D or signature_Fh == V_CG2_2D or
             signature_Fh == V_CG3_2D or signature_Fh == V_CG4_2D or
             signature_Fh == V_CG1_3D or signature_Fh == V_CG2_3D or
             signature_Fh == V_CG3_3D or signature_Fh == V_CG4_3D) {
    if (D == 2) {

      if (signature == V_CG1_2D) {
        a_ = std::make_shared<PDESmoothVCGh_2D_P1::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_2D_P1::LinearForm>(V_);
      } else if (signature == V_CG2_2D) {
        a_ = std::make_shared<PDESmoothVCGh_2D_P2::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_2D_P2::LinearForm>(V_);
      } else if (signature == V_CG3_2D) {
        a_ = std::make_shared<PDESmoothVCGh_2D_P3::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_2D_P3::LinearForm>(V_);
      }

      else if (signature == Q_CG1_2D) {
        a_ = std::make_shared<PDESmoothVQCGh_2D_P1::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_2D_P1::LinearForm>(V_);
      } else if (signature == Q_CG2_2D) {
        a_ = std::make_shared<PDESmoothVQCGh_2D_P2::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_2D_P2::LinearForm>(V_);
      } else if (signature == Q_CG3_2D) {
        a_ = std::make_shared<PDESmoothVQCGh_2D_P3::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_2D_P3::LinearForm>(V_);
      }
    } else if (D == 3) {
      if (signature == V_CG1_3D) {
        a_ = std::make_shared<PDESmoothVCGh_3D_P1::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_3D_P1::LinearForm>(V_);
      } else if (signature == V_CG2_3D) {
        a_ = std::make_shared<PDESmoothVCGh_3D_P2::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_3D_P2::LinearForm>(V_);
      } else if (signature == V_CG3_3D) {
        a_ = std::make_shared<PDESmoothVCGh_3D_P3::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVCGh_3D_P3::LinearForm>(V_);
      } else if (signature == Q_CG1_3D) {
        a_ = std::make_shared<PDESmoothVQCGh_3D_P1::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_3D_P1::LinearForm>(V_);
      } else if (signature == Q_CG2_3D) {
        a_ = std::make_shared<PDESmoothVQCGh_3D_P2::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_3D_P2::LinearForm>(V_);
      } else if (signature == Q_CG3_3D) {
        a_ = std::make_shared<PDESmoothVQCGh_3D_P3::BilinearForm>(V_, V_);
        L_ = std::make_shared<PDESmoothVQCGh_3D_P3::LinearForm>(V_);
      } else {
        error("Function spaces for P%d does not exist", pol_degree);
      }
    } else {
      error("Function spaces for P%d does not exist", pol_degree);
    }
  } else if (signature_Fh == V_DG0_2D) {
    if (signature == V_CG1_2D) {
      a_ = std::make_shared<PDESmoothV_2D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothV_2D_P1::LinearForm>(V_);
    }
  } else if (signature == Q_CG1_3D) {
    if (signature_Fh == Q_CG1_3D) {
      // smoothing using CG mesh-size
      a_ = std::make_shared<PDESmoothCGh_3D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_3D_P1::LinearForm>(V_);
    }
  }

  else if (signature == Q_CG2_3D) {
    if (signature_Fh == Q_CG2_3D) {
      // smoothing using CG mesh-size
      a_ = std::make_shared<PDESmoothCGh_3D_P2::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_3D_P2::LinearForm>(V_);
    }
  }

  else if (signature == Q_CG3_3D) {
    if (signature_Fh == Q_CG3_3D) {
      // smoothing using CG mesh-size
      a_ = std::make_shared<PDESmoothCGh_3D_P3::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothCGh_3D_P3::LinearForm>(V_);
    }
  }

  else if (signature_Fh == Q_DG0_3D) {
    // smoothing using DG mesh-size
    a_ = std::make_shared<PDESmooth_3D_P1::BilinearForm>(V_, V_);
    L_ = std::make_shared<PDESmooth_3D_P1::LinearForm>(V_);
  }

  else if (signature == V_CG1_3D) {
    a_ = std::make_shared<PDESmoothV_3D_P1::BilinearForm>(V_, V_);
    L_ = std::make_shared<PDESmoothV_3D_P1::LinearForm>(V_);
  } else if (signature_Fh == V_CG1_3D or signature_Fh == V_CG2_3D or
             signature_Fh == V_CG3_3D or signature_Fh == V_CG4_3D) {
    if (signature == V_CG1_3D) {
      a_ = std::make_shared<PDESmoothVCGh_3D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCGh_3D_P1::LinearForm>(V_);
    } else if (signature == V_CG2_3D) {
      a_ = std::make_shared<PDESmoothVCGh_3D_P2::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCGh_3D_P2::LinearForm>(V_);
    } else if (signature == V_CG3_3D) {
      a_ = std::make_shared<PDESmoothVCGh_3D_P3::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVCGh_3D_P3::LinearForm>(V_);
    } else if (signature == Q_CG1_3D) {
      a_ = std::make_shared<PDESmoothVQCGh_3D_P1::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVQCGh_3D_P1::LinearForm>(V_);
    } else if (signature == Q_CG2_3D) {
      a_ = std::make_shared<PDESmoothVQCGh_3D_P2::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVQCGh_3D_P2::LinearForm>(V_);
    } else if (signature == Q_CG3_3D) {
      a_ = std::make_shared<PDESmoothVQCGh_3D_P3::BilinearForm>(V_, V_);
      L_ = std::make_shared<PDESmoothVQCGh_3D_P3::LinearForm>(V_);
    } else {
      error("Function spaces for P%d does not exist", pol_degree);
    }
  }

  a_->set_coefficient("h", Fh_);
  a_->set_coefficient("weight", Fweight);
  assemble(A_, *a_);
  solver_mass_ = std::make_shared<LinearSolver>("gmres");
  solver_mass_->parameters["relative_tolerance"] = 1.0e-13;
  solver_mass_->parameters["absolute_tolerance"] = 1.0e-20;
}

// -----------------------------------------------------------------------------
void PDESmooth::smooth(std::shared_ptr<Function> Fw) {
  double const smoothing_weight = parameters["smoothing_weight"];

  if (smoothing_weight < 1e-5) {
    return;
  }

  L_->set_coefficient("u", Fw);

  assemble(b_, *L_);
  if (bcs_ != NULL) {
    for (std::size_t i = 0; i < bcs_->size(); ++i) {
      (*bcs_)[i]->apply(A_, b_);
    }
  }
  solver_mass_->solve(A_, *Fw->vector(), b_);
}

// -----------------------------------------------------------------------------
void PDESmooth::smooth(std::shared_ptr<Function> Fw, double const &weight) {
  Constant Fweight(weight);
  a_->set_coefficient("weight",
                      std::shared_ptr<const GenericFunction>(&Fweight));
  assemble(A_, *a_);

  L_->set_coefficient("u", Fw);
  assemble(b_, *L_);
  if (bcs_ != NULL) {
    for (std::size_t i = 0; i < bcs_->size(); ++i) {
      (*bcs_)[i]->apply(A_, b_);
    }
  }
  solver_mass_->solve(A_, *Fw->vector(), b_);
}

void PDESmooth::update_weight(double const &weight) {
  auto Fweight = std::make_shared<const Constant>(weight);
  a_->set_coefficient("weight", Fweight);
  assemble(A_, *a_);
}
