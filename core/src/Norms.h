// Copyright (C) 2016 Murtazo Nazarov
// Licensed under the MIT License – see the LICENSE file for details
// 
//
// First added:  2016-02-14
// Last changed: 2019-12-22


#ifndef __NORMS_H
#define __NORMS_H

#include <dolfin/function/Function.h>
#include <dolfin/function/GenericFunction.h>
#include <memory>

///
namespace dolfin
{
  class Function;
  class GenericFunction;
  class Mesh;
  class FunctionSpace;
  class Form;
}

using dolfin::Function;
using dolfin::GenericFunction;
using dolfin::FunctionSpace;
using dolfin::Form;
using dolfin::Mesh;

namespace fem_varden_flow
{

class Norms
{
 public:
  
  ///
  static void compute_error(Function const& Fu, GenericFunction const& Fu_ex,
			    std::string const& file_name);
  ///
  static void compute_absolute_error(Function const& Fu, GenericFunction const& Fu_ex,
			    std::string const& file_name);
  ///
  static void get_forms(std::shared_ptr<const Mesh> mesh_,
			std::shared_ptr<Form>& L1, std::shared_ptr<Form>& L2,
			std::shared_ptr<FunctionSpace>& V_ex,
			std::size_t const pol_degree, std::size_t const value_size);
};
}

#endif
