// Copyright (C) 2023 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details

#include <stdexcept>
// dolfin
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/fem/assemble.h>
#include <dolfin/fem/Form.h>
#include <dolfin/log/log.h>
#include <dolfin/function/Expression.h>

// this demo
#include "../ufl/V_P3.h"
#include "../ufl/V_P2.h"
#include "../ufl/V_P1.h"
#include "../ufl/Q_P3.h"
#include "../ufl/Q_P2.h"
#include "../ufl/Q_P1.h"

// Define domains for marking
class LeftUnitDomain : public dolfin::SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return 
      on_boundary && dolfin::near(x[0], 0);
  }
};

class RightUnitDomain : public dolfin::SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return 
      on_boundary && dolfin::near(x[0], 1.0);
  }
};

class Tracer : public dolfin::Expression
{
public:
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
  }
};

class Velocity : public dolfin::Expression
{
public:
	Velocity() : Expression(2) {}
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
		values[1] = x[1];
  }
};

class FEMInfo
{
public:
	/// Constructor
	FEMInfo(std::shared_ptr<dolfin::Mesh> mesh, std::size_t cg_order)
	{
		dolfin_assert(mesh->geometry().dim() == 2);
		V_CG1 = std::make_shared<V_P1::FunctionSpace>(mesh);
		switch (cg_order)
			{
			case 1:
				std::cout << "Using CG1 space.\n";
				V = std::make_shared<V_P1::FunctionSpace>(mesh);
				Q = std::make_shared<Q_P1::FunctionSpace>(mesh);
				break;
			case 2:
				std::cout << "Using CG2 space.\n";
				V = std::make_shared<V_P2::FunctionSpace>(mesh);
				Q = std::make_shared<Q_P2::FunctionSpace>(mesh);
				break;
			case 3:
				std::cout << "Using CG3 space.\n";
				V = std::make_shared<V_P3::FunctionSpace>(mesh);
				Q = std::make_shared<Q_P3::FunctionSpace>(mesh);
				break;
			default:
				throw std::invalid_argument("Incorrect order of 2D function space");
			}
		
	}
	
	/// Public members
	std::shared_ptr<dolfin::FunctionSpace> V;
	std::shared_ptr<dolfin::FunctionSpace> V_CG1;
	std::shared_ptr<dolfin::FunctionSpace> Q;
};
