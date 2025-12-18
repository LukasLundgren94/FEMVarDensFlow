// Copyright (c) 2018 Murtazo Nazarov
// Copyright (c) 2019-2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <dolfin/fem/Form.h>
#include <dolfin/la/LinearSolver.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/SubDomain.h>

#include "VD_NSEManifoldEquation.h"

#include "ufl/Manifold_2D_P2P1.h"
#include "ufl/Manifold_2D_P1P1.h"
#include "ufl/Manifold_2D_P3P3.h"
#include "ufl/Manifold_2D_P3P2.h"
#include "ufl/Manifold_2D_P4P3.h"
#include "ufl/Manifold_3D_P3P2.h"
#include "ufl/Manifold_3D_P2P1.h"
#include "ufl/Manifold_3D_P1P1.h"
#include "ufl/Manifold_3D_P3P3.h"
#include "ufl/Manifold_3D_P4P3.h"

#include "System.h"

using namespace fem_varden_flow;
using namespace dolfin;

//-----------------------------------------------------------------------------
void VD_NSEManifoldEquation::init(std::shared_ptr<const BoundaryMesh> mesh,
                                   System &eq, Parameters &p) {
  const std::size_t D = mesh->topology().dim() + 1; // boundary mesh is one dimension lower
  int const pol_degree_V_M = p["PolDegree_V_M"];
  int const pol_degree_Q = p["PolDegree_Q"];

  if (D == 2)
  {
    if ( ( pol_degree_V_M == 3) && ( pol_degree_Q == 2) ) 
      {
        // if (!eq.M) 
          eq.M = std::make_shared<Manifold_2D_P3P2::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_2D_P3P2::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_2D_P3P2::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 2) && ( pol_degree_Q == 1) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_2D_P2P1::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_2D_P2P1::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_2D_P2P1::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 1) && ( pol_degree_Q == 1) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_2D_P1P1::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_2D_P1P1::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_2D_P1P1::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 3) && ( pol_degree_Q == 3) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_2D_P3P3::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_2D_P3P3::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_2D_P3P3::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 4) && ( pol_degree_Q == 3) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_2D_P4P3::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_2D_P4P3::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_2D_P4P3::Form_L>(eq.M); 
      }
  }

  if (D == 3)
  {
    if ( ( pol_degree_V_M == 3) && ( pol_degree_Q == 2) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_3D_P3P2::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_3D_P3P2::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_3D_P3P2::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 2) && ( pol_degree_Q == 1) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_3D_P2P1::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_3D_P2P1::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_3D_P2P1::Form_L>(eq.M);  
      }
    if ( ( pol_degree_V_M == 1) && ( pol_degree_Q == 1) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_3D_P1P1::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_3D_P1P1::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_3D_P1P1::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 3) && ( pol_degree_Q == 3) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_3D_P3P3::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_3D_P3P3::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_3D_P3P3::Form_L>(eq.M); 
      }
    if ( ( pol_degree_V_M == 4) && ( pol_degree_Q == 3) ) 
      {
        if (!eq.M) 
          eq.M = std::make_shared<Manifold_3D_P4P3::FunctionSpace>(mesh); 
       eq.a = std::make_shared<Manifold_3D_P4P3::Form_a>(eq.M, eq.M); 
       eq.L = std::make_shared<Manifold_3D_P4P3::Form_L>(eq.M); 
      }
  }

}

//-----------------------------------------------------------------------------
void VD_NSEManifoldEquation::init(std::shared_ptr<const BoundaryMesh> mesh,
                                   std::shared_ptr<const FunctionSpace> M,
                                   System &eq, Parameters &p) {
  const std::size_t D = mesh->topology().dim();

  eq.M = M;
  VD_NSEManifoldEquation::init(mesh, eq, p);
}
