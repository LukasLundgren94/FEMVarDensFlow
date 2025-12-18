// Copyright (C) 2023 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/generation/RectangleMesh.h>
#include <dolfin/generation/UnitSquareMesh.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/la/solve.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/refinement/refine.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/dolfin_parameter.h>
#include <dolfin/log/log.h>
#include <dolfin/common/SubSystemsManager.h>

/// core
// #include <SlipBC.h>
// #include <NodeNormal.h>

#include "ocean/meltparameterization/ExchangeVelocity.h"

  
void print_gammas(fem_varden_flow::ExchangeVelocity e, std::vector<double> u_stars)
{
  std::cout << "gammaT and gammaS values (see Holland/Jenkins Fig 4)\n"
  << "=======================================================\n";
  std::cout << "ExchangeVelocity with method: " << e.get_method() << "\n";
  double gammaT, gammaS;
  for (auto& u_star: u_stars)
    {
      gammaT = e.gammaT(u_star);
      gammaS = e.gammaS(u_star);
      std::cout << "u_star:" << u_star << " gammaT: " << gammaT
      << " gammaS: " << gammaS << " gammaS/gammaT: " << gammaS/gammaT << "\n";
    }
  std::cout << "\n";
}

int main(int argc, char *argv[]) {
  // Necesarry if no MPI initialization (e.g., creating mesh, function
  // spaces etc) has been done
  dolfin::SubSystemsManager::init_mpi();

  // Vector of "precalculated" u_star velocities in typical range (?)
  // from Figure 4 Jenkins/Holland (1999).
  std::vector<double> u_stars{0.001, 0.005, 0.01, 0.015, 0.02};
  
  // Try the different methods with default settings
  fem_varden_flow::ExchangeVelocity e_full;
  fem_varden_flow::ExchangeVelocity e_simple("simple");
  fem_varden_flow::ExchangeVelocity e_linear("linear");

  // Print results
  print_gammas(e_full, u_stars);
  print_gammas(e_simple, u_stars);
  print_gammas(e_linear, u_stars);
  
  // Change some parameters and update.
  dolfin::Parameters p("local_parameters");
  // add f == 0: should compute without coriolis and then be the same
  // Coriolis parameter
  p.add("coriolis_parameter_exchange", 0.0);
  // Update parameters
  e_full.update_parameters(p);
  std::cout << "Updated parameters to:\n";
  dolfin::info(e_full.parameters, true);
  std::cout <<
  "Results should be same as 'simple' method with default params:\n";
  print_gammas(e_full, u_stars);
  
  // Coriolis parameter
  p["coriolis_parameter_exchange"] = -1e-4; // back to default value
  // Von Kàrmàn's constant
  p.add("von_karman_constant", 0.45);          // default: 0.4
  // Prandtl number
  p.add("prandtl_number_exchange", 14.0);         // default 13.8
  // non-existent parameter: will be ignored
  p.add("fucker", 666.0);
  // Update parameters
  e_full.update_parameters(p);
  std::cout << "Updated parameters to:\n";
  dolfin::info(e_full.parameters, true);
  print_gammas(e_full, u_stars);

  // Reupdate with default parameters
  e_full.update_parameters(e_full.default_parameters());
  std::cout << "Updated parameters to default parameters:\n";
  dolfin::info(e_full.parameters, true);
  std::cout <<
  "Results should be same as initial 'full' method:\n";
  print_gammas(e_full, u_stars);
  
  // fem_varden_flow::ExchangeVelocity e1(p);
  // std::cout << "ExchangeVelocity with method: " << e1.get_method() << "\n";
  // dolfin::info(e1.parameters, true);
  // p["k"] = 666.0;
  // e1.update_parameters(p);
  // dolfin::info(e1.parameters, true);
  
  // p["xi_N"] = 123246.0;
  // // parameter to be ignored
  // p.add("fuckface", 32);
  // fem_varden_flow::ExchangeVelocity e2(p, "simple");
  // std::cout << "ExchangeVelocity with method: " << e2.get_method() << "\n";
  // dolfin::info(e2.parameters, true);

    
  // invalid method
  // fem_varden_flow::ExchangeVelocity e4(p, "fucker");
}
