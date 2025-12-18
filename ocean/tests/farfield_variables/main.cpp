// Copyright (C) 2023 Christian Helanow
// Licensed under the MIT License – see the LICENSE file for details

// Just a small test for the values of far-field variables

/// dolfin
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
// #include <dolfin/generation/RectangleMesh.h>
#include <dolfin/generation/UnitSquareMesh.h>
#include <dolfin/generation/UnitCubeMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/DomainBoundary.h>
// #include <dolfin/la/solve.h>
// #include <dolfin/mesh/CellType.h>
// #include <dolfin/parameter/GlobalParameters.h>
// #include <dolfin/refinement/refine.h>
#include <dolfin/io/XDMFFile.h>
// #include <dolfin/common/MPI.h>
#include <dolfin/parameter/dolfin_parameter.h>
#include <dolfin/log/log.h>
#include <dolfin/math/basic.h>
// #include <dolfin/common/MPI.h>
#include <dolfin/common/Array.h>
// #include <dolfin/common/types.h>

/// general
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

// ocean
#include <ocean/common/UnitNodeNormal.h>
#include <ocean/common/UnitSlipBC.h>
#include <ocean/meltparameterization/FarFieldTracer.h>
#include <ocean/meltparameterization/FrictionVelocity.h>

// this demo/test
#include "farfield_variables.h"


int main(int argc, char *argv[]) {
		// allow extrapolation in parallel
 	dolfin::parameters["std_out_all_processes"] = false;
	// Create mesh and define function space
	auto mesh = std::make_shared<dolfin::UnitSquareMesh>(8, 8);
	// auto mesh = std::make_shared<dolfin::Mesh>();
	// {
	//   dolfin::XDMFFile file("../meshes/ryder.xdmf");
	//   file.read(*mesh);
	// }
	dolfin::parameters["allow_extrapolation"] = true;




	
	FEMInfo feminfo(mesh, 2);
	auto V = feminfo.V;
	auto V_CG1 = feminfo.V_CG1;
	auto Q = feminfo.Q;
	
	// // Define Function f or solution
	bool normalize{true};
	fem_varden_flow::UnitNodeNormal node_normal(V, V_CG1, normalize);
	// dolfin::XDMFFile xdmf_normal(mesh->mpi_comm(),
	//                            std::string("results/normal.xdmf"));
	// xdmf_normal.write_checkpoint(node_normal.normal(), "Normal");
	// xdmf_normal.close();

	// make up random function for testing
	std::shared_ptr<dolfin::Function> vel = std::make_shared<dolfin::Function>(V);
	std::shared_ptr<dolfin::Function> tracer = std::make_shared<dolfin::Function>(Q);
	vel->set_allow_extrapolation(true);
	vel->interpolate(Velocity());
	tracer->set_allow_extrapolation(true);
	tracer->interpolate(Tracer());

	dolfin::XDMFFile xdmf_tracer(mesh->mpi_comm(),
								 std::string("results/tracer.xdmf"));
	xdmf_tracer.write_checkpoint(*tracer, "tracer");
	xdmf_tracer.close();
	dolfin::XDMFFile xdmf_vel(mesh->mpi_comm(),
								 std::string("results/velocity.xdmf"));
	xdmf_vel.write_checkpoint(*vel, "velocity");
	xdmf_vel.close();

	// get farfield variables change the boundary layer thickness for
	// this problem by updating local parameter sets
	dolfin::Parameters farfield_params = fem_varden_flow::FarFieldTracer::default_parameters();
	farfield_params["farfield_boundary_layer_thickness"] = 0.1;
	fem_varden_flow::FarFieldTracer ff_tracer(tracer, node_normal, farfield_params);

	dolfin::Parameters friction_params = fem_varden_flow::FrictionVelocity::default_parameters();
	friction_params["farfield_boundary_layer_thickness"] = 0.1;
	friction_params["minimum_farfield_velocity"] = farfield_params["minimum_farfield_velocity"];
	friction_params["farfield_drag_coefficient"] = farfield_params["farfield_drag_coefficient"];
	friction_params["farfield_roughness_length"] = farfield_params["farfield_roughness_length"];
	fem_varden_flow::FrictionVelocity friction_vel(vel, node_normal, friction_params);
	// ff_tracer.set_allow_extrapolation(true);
	// friction_vel.set_allow_extrapolation(true);
	
	
	// Mark domain
	auto markers = std::make_shared<MeshFunction<std::size_t>>
		(mesh, mesh->topology().dim()-1, 0);
	auto left_domain = std::make_shared<LeftUnitDomain>();
	auto right_domain = std::make_shared<RightUnitDomain>();
	left_domain->mark(*markers, 1);
	right_domain->mark(*markers, 2);
	// collect boundary dofs for each function space
	std::unordered_set<std::size_t> boundary_dofs_v;
	std::unordered_set<std::size_t> boundary_dofs_q;
	for (std::size_t i = 1; i <= 2; ++i){
		dolfin::DirichletBC bcV(V, std::make_shared<dolfin::Constant>(0.0, 0.0),
								markers, i);
		dolfin::DirichletBC bcQ(Q, std::make_shared<dolfin::Constant>(0.0),
								markers, i);
		dolfin::DirichletBC::Map boundary_vals_v;
		dolfin::DirichletBC::Map boundary_vals_q;
		bcV.get_boundary_values(boundary_vals_v);
		bcQ.get_boundary_values(boundary_vals_q);
		boundary_dofs_v.reserve(boundary_vals_v.size());
		boundary_dofs_q.reserve(boundary_vals_q.size());
		for (auto iter : boundary_vals_v)
		{
			boundary_dofs_v.insert(iter.first);
		}
		for (auto iter : boundary_vals_q)
		{
			boundary_dofs_q.insert(iter.first);
		}
	}
	std::cout << "TRACER FAR-FIELD VALUES, DOFS AND COORS:::\n";
	double blt = ff_tracer.parameters["farfield_boundary_layer_thickness"];
	std::cout << "Boundary layer thickness: " << blt  << "\n";

	std::vector<double> Q_dof_coors = Q->tabulate_dof_coordinates();
	dolfin::Array<double> coors(2);
	dolfin::Array<double> tracer_value(1);
	dolfin::Array<double> ff_tracer_value(1);
	{
		using namespace std;
		for (auto ind : boundary_dofs_q)
		{
			coors[0] = Q_dof_coors[ind * 2];
			coors[1] = Q_dof_coors[ind * 2 + 1];
			tracer->eval(tracer_value, coors);
			ff_tracer.eval(ff_tracer_value, coors);
			std::cout
				<< "Dof: " << setw(6) << setprecision(5) << ind
				<< " Farfield tracer value: " << setw(6) << setprecision(5) << fixed
				<<  ff_tracer_value[0]
				<< " Tracer value: " <<  setw(6) << setprecision(5) << fixed
				<< tracer_value[0]
				<< " Coordinate: (" <<  setw(6) << setprecision(5) << fixed
				<< coors[0]
				<< ", " <<  setw(6) << setprecision(5) << fixed
				<< coors[1] << ")\n";
			
		}
	}

	std::cout << "FRICTION VELOCITY VALUES, DOFS AND COORS:::\n";
	std::cout << "(Lots of doubles, due to dofs for each vel component gets a coordinate)\n";
	std::cout << "Boundary layer thickness: " << blt  << "\n";

	std::vector<double> V_dof_coors = V->tabulate_dof_coordinates();
	dolfin::Array<double> vel_value(2);
	dolfin::Array<double> friction_value(1);
	{
		using namespace std;
		for (auto ind : boundary_dofs_v)
		{
			coors[0] = V_dof_coors[ind * 2];
			coors[1] = V_dof_coors[ind * 2 + 1];
			vel->eval(vel_value, coors);
			friction_vel.eval(friction_value, coors);
			std::cout
				<< "Dof: " << setw(6) << setprecision(5) << ind
				<< " Friction velocity value: " << setw(6) << setprecision(5) << fixed
				<<  friction_value[0]
				<< " Velocity value: ( " <<  setw(6) << setprecision(5) << fixed
				<< vel_value[0]
				<< ", " <<  setw(6) << setprecision(5) << fixed
				<< vel_value[1] << ") "
				<< " Coordinate: (" <<  setw(6) << setprecision(5) << fixed
				<< coors[0]
				<< ", " <<  setw(6) << setprecision(5) << fixed
				<< coors[1] << ")\n";
			
		}
	}

}
