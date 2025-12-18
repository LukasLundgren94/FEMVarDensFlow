// Copyright (C) 2016 Murtazo Nazarov
// Licensed under the MIT License – see the LICENSE file for details
// 
// 
// 
//
// First added:  2016-02-03
// Last changed: 2019-12-22

#ifndef __PDESMOOTH_H
#define __PDESMOOTH_H

#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/parameter/Parameter.h>

///
namespace dolfin
{
  class Function;
  class Matrix;
  class Vector;
  class LinearSolver;
  class Variable;
}

using dolfin::Function;
using dolfin::Matrix;
using dolfin::Vector;
using dolfin::LinearSolver;
using dolfin::DirichletBC;
using dolfin::FunctionSpace;
using dolfin::Form;
using dolfin::Parameters;

#include "ufl/pdesmooth/PDESmooth_2D_P1.h"
#include "ufl/pdesmooth/PDESmoothCGh_2D_P1.h"
#include "ufl/pdesmooth/PDESmoothV_2D_P1.h"
#include "ufl/pdesmooth/PDESmoothVCGh_2D_P1.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_2D_P1.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_2D_P1.h"

#include "ufl/pdesmooth/PDESmooth_2D_P2.h"
#include "ufl/pdesmooth/PDESmoothCGh_2D_P2.h"
#include "ufl/pdesmooth/PDESmoothV_2D_P2.h"
#include "ufl/pdesmooth/PDESmoothVCGh_2D_P2.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_2D_P2.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_2D_P2.h"

#include "ufl/pdesmooth/PDESmooth_2D_P3.h"
#include "ufl/pdesmooth/PDESmoothCGh_2D_P3.h"
#include "ufl/pdesmooth/PDESmoothV_2D_P3.h"
#include "ufl/pdesmooth/PDESmoothVCGh_2D_P3.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_2D_P3.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_2D_P3.h"

/* /// 3D */
#include "ufl/pdesmooth/PDESmooth_3D_P1.h"
#include "ufl/pdesmooth/PDESmoothCGh_3D_P1.h"
#include "ufl/pdesmooth/PDESmoothV_3D_P1.h"
#include "ufl/pdesmooth/PDESmoothVCGh_3D_P1.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_3D_P1.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_3D_P1.h"

#include "ufl/pdesmooth/PDESmooth_3D_P2.h"
#include "ufl/pdesmooth/PDESmoothCGh_3D_P2.h"
#include "ufl/pdesmooth/PDESmoothV_3D_P2.h"
#include "ufl/pdesmooth/PDESmoothVCGh_3D_P2.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_3D_P2.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_3D_P2.h"

#include "ufl/pdesmooth/PDESmooth_3D_P3.h"
#include "ufl/pdesmooth/PDESmoothCGh_3D_P3.h"
#include "ufl/pdesmooth/PDESmoothV_3D_P3.h"
#include "ufl/pdesmooth/PDESmoothVCGh_3D_P3.h"
#include "ufl/pdesmooth/PDESmoothVQCGh_3D_P3.h"
#include "ufl/pdesmooth/PDESmoothVCG1h_3D_P3.h"

namespace fem_varden_flow
{

class PDESmooth: public Variable
{
public:
  ///
  PDESmooth(std::shared_ptr<const FunctionSpace> V,
            std::shared_ptr<const Function> Fh, Parameters& p);

    PDESmooth(std::shared_ptr<const FunctionSpace> V,
            std::shared_ptr<const Function> Fh, Parameters& p, std::shared_ptr<std::vector<DirichletBC *>> bcs);

  ///
  ~PDESmooth();

  ///
  void smooth(std::shared_ptr<Function> Fw);

  ///
  void smooth(std::shared_ptr<Function> Fw, double const& weight);

  void update_weight(double const& weight);

private:
  std::shared_ptr<const FunctionSpace> V_;
  std::shared_ptr<const Function> Fh_;
  std::shared_ptr<Form> a_;
  std::shared_ptr<Form> L_;
  std::shared_ptr<const std::vector<DirichletBC*>> bcs_;


  std::shared_ptr<LinearSolver> solver_mass_;
  Matrix A_;
  Vector b_;

  void init();

  ///
  std::string Q_CG1_2D = "FiniteElement('Lagrange', triangle, 1)";
  std::string Q_CG2_2D = "FiniteElement('Lagrange', triangle, 2)";
  std::string Q_CG3_2D = "FiniteElement('Lagrange', triangle, 3)";
  std::string Q_CG4_2D = "FiniteElement('Lagrange', triangle, 4)";

  ///
  std::string V_CG1_2D = "VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2)";
  std::string V_CG2_2D = "VectorElement(FiniteElement('Lagrange', triangle, 2), dim=2)";
  std::string V_CG3_2D = "VectorElement(FiniteElement('Lagrange', triangle, 3), dim=2)";
  std::string V_CG4_2D = "VectorElement(FiniteElement('Lagrange', triangle, 4), dim=2)";

  ///
  std::string Q_CG1_3D = "FiniteElement('Lagrange', tetrahedron, 1)";
  std::string Q_CG2_3D = "FiniteElement('Lagrange', tetrahedron, 2)";
  std::string Q_CG3_3D = "FiniteElement('Lagrange', tetrahedron, 3)";
  std::string Q_CG4_3D = "FiniteElement('Lagrange', tetrahedron, 4)";

  ///
  std::string V_CG1_3D = "VectorElement(FiniteElement('Lagrange', tetrahedron, 1), dim=3)";
  std::string V_CG2_3D = "VectorElement(FiniteElement('Lagrange', tetrahedron, 2), dim=3)";
  std::string V_CG3_3D = "VectorElement(FiniteElement('Lagrange', tetrahedron, 3), dim=3)";
  std::string V_CG4_3D = "VectorElement(FiniteElement('Lagrange', tetrahedron, 4), dim=3)";

  ///
  std::string Q_DG0_2D = "FiniteElement('Discontinuous Lagrange', triangle, 0)";
  std::string V_DG0_2D = "VectorElement(FiniteElement('Discontinuous Lagrange', triangle, 0), dim=2)";

  ///
  std::string Q_DG0_3D = "FiniteElement('Discontinuous Lagrange', tetrahedron, 0)";
  std::string V_DG0_3D = "VectorElement(FiniteElement('Discontinuous Lagrange', tetrahedron, 0), dim=3)";

  const std::string SpaceTable_2D[10] = {
       Q_CG1_2D, Q_CG2_2D, Q_CG3_2D, Q_CG4_2D,
       V_CG1_2D, V_CG2_2D, V_CG3_2D, V_CG4_2D,
       Q_DG0_2D, V_DG0_2D
       };
};
}

#endif

