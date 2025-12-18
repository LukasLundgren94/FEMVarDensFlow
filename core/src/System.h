// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2021-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details
//

#ifndef __SYSTEM_H
#define __SYSTEM_H

#include <string>
#include <dolfin/function/Function.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/fem/Form.h>
#include <dolfin/common/Variable.h>

///
namespace dolfin
{
  class FunctionSpace;
  class Function;
  class Constant;
}

///
using dolfin::Variable;
using dolfin::Parameters;
using dolfin::Matrix;
using dolfin::Vector;
using dolfin::FunctionSpace;
using dolfin::Form;
using dolfin::LinearSolver;


namespace fem_varden_flow
{
class System : public Variable
{
public:
  ///
  System();

  ////
  ~System();

  ///
  std::shared_ptr<const FunctionSpace> W;
  std::shared_ptr<const FunctionSpace> Q;
  std::shared_ptr<const FunctionSpace> V;
  std::shared_ptr<const FunctionSpace> M;
  std::shared_ptr<const FunctionSpace> DG0;
  std::shared_ptr<const FunctionSpace> V_DG0;
  std::shared_ptr<const FunctionSpace> W_CG1;
  
  ///
  std::shared_ptr<Form> a;
  std::shared_ptr<Form> L;

  ///
  std::shared_ptr<Form> F1;
  std::shared_ptr<Form> F2;

  ///
  Matrix A;
  Vector b;

  ///
  std::shared_ptr<LinearSolver> ls;

  void init();
};
}
#endif
