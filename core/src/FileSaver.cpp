// Copyright (C) 2015 Murtazo Nazarov
// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
//

#include <dolfin/common/ArrayView.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/LagrangeInterpolator.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>

#include "FileSaver.h"

///
namespace dolfin {
class Expression;
}

///
using dolfin::Array;
using dolfin::error;
using dolfin::Expression;
using dolfin::info;
using dolfin::LagrangeInterpolator;
using dolfin::set_log_active;
using dolfin::warning;

using namespace fem_varden_flow;

//-------------------------------------------------------------------------------
FileSaver::FileSaver(double const &time, Parameters const &parameters)
    : time_(time), parameters_(parameters) {
  sample_ = 0;
  sample_cp_ = 0;
  folder << "results/";
  init();
}

//-------------------------------------------------------------------------------
void FileSaver::update() { init(); }

//-------------------------------------------------------------------------------
void FileSaver::init() {
  if (parameters_.has_parameter("output_file_type"))
    s_file_type_.assign(parameters_["output_file_type"]);
  else
    s_file_type_.assign("vtk");

  Time_ = parameters_["T"];

  if (!parameters_.has_parameter("t_save")) {
    error("No t_save specified");
  } else {
    t_save_ = parameters_["t_save"];
  }

  t_save_cp_ = parameters_["t_save_checkpoint"];

  /// shift the t_save_ if T is a checkpoint
  // modified by Lukas (1 +) so that we don't overwrite old checkpoint and
  // solution
  if (time_ > 0) {
    sample_ += 1 + static_cast<int>(std::floor(time_ / t_save_));
    sample_cp_ += 1 + static_cast<int>(std::floor(time_ / t_save_cp_));
  }
}

//-------------------------------------------------------------------------------
FileSaver::~FileSaver() {
  files_.clear();
  functions_.clear();
  functions_fine_.clear();
  files_.clear();
  xfiles_.clear();
  xfiles_cp_.clear();
  function_names_.clear();
  functions_cp_.clear();
}

//-------------------------------------------------------------------------------
void FileSaver::add(const Function &u, const std::string &name) {
  functions_.push_back(&u);

  std::ostringstream ext;
  ext << "_pol_" << u.function_space()->element()->ufc_element()->degree()
      << "_vert_" << u.function_space()->dim() << "_";

  if (s_file_type_ == "vtk") {
    file_ = new File(folder.str() + name + ext.str() + ".pvd", "compressed");
    files_.push_back(file_);
  } else if (s_file_type_ == "xdmf") {
    xfile_ = new XDMFFile(folder.str() + name + ext.str() + ".xdmf");
    xfile_->parameters["flush_output"] = true;
    xfiles_.push_back(xfile_);
  }
}

//-------------------------------------------------------------------------------
void FileSaver::add(const Function &u,
                    std::shared_ptr<FunctionSpace> V_CG1_fine,
                    const std::string &name) {
  const Function u_CG1_fine(V_CG1_fine);
  // std::pair<const Function*, const Function*> pair_functions;
  // pair_functions.first = &u;
  // pair_functions.second = &u_CG1_fine;
  std::pair<const Function *, std::shared_ptr<const FunctionSpace>>
      pair_functions(&u, V_CG1_fine);
  functions_P_.push_back(pair_functions);

  std::ostringstream ext;
  ext << "_pol_" << u.function_space()->element()->ufc_element()->degree()
      << "_vert_" << u.function_space()->dim() << "_";

  if (s_file_type_ == "vtk") {
    file_ = new File(folder.str() + "fine/" + name + ext.str() + ".pvd",
                     "compressed");
    files_fine_.push_back(file_);
  } else if (s_file_type_ == "xdmf") {
    xfile_ = new XDMFFile(folder.str() + "fine/" + name + ext.str() + ".xdmf");
    xfile_->parameters["flush_output"] = true;
    xfiles_fine_.push_back(xfile_);
  }
}

//-------------------------------------------------------------------------------
void FileSaver::save() {
  LagrangeInterpolator Lin;

  if ((files_.size() == 0 && xfiles_.size() == 0) &&
      (files_fine_.size() == 0 && xfiles_fine_.size() == 0)) {
    warning("Nothing to save");
    return;
  }

  if ((time_ < eps_) || (time_ > t_save_ * sample_) ||
      (fabs(time_ - Time_) < eps_)) {
    set_log_active(true);
    info("Saving %s output to file at t = %f, sample = %d",
         s_file_type_.c_str(), time_, sample_);
    set_log_active(false);

    if (s_file_type_ == "vtk") {
      /// save
      for (std::size_t i = 0; i < files_.size(); ++i)
        *files_[i] << std::pair<const Function *, double>(&((*functions_[i])),
                                                          time_);

      /// save fine P1 solutions
      for (std::size_t i = 0; i < files_fine_.size(); ++i) {
        // Function F_fine(*(functions_P_[i].second));
        Function F_fine(functions_P_[i].second);
        Lin.interpolate(F_fine, *(functions_P_[i].first));
        *files_fine_[i] << std::pair<const Function *, double>(&(F_fine),
                                                               time_);
      }

    } else if (s_file_type_ == "xdmf") {
      for (std::size_t i = 0; i < xfiles_.size(); ++i)
        (*xfiles_[i]).write(*functions_[i], time_);

      /// save fine P1 solutions
      for (std::size_t i = 0; i < xfiles_fine_.size(); ++i) {
        Function F_fine(functions_P_[i].second);
        Lin.interpolate(F_fine, *(functions_P_[i].first));
        (*xfiles_fine_[i]).write(F_fine, time_);
      }
    }
    sample_++;
  }
}

//-------------------------------------------------------------------------------
void FileSaver::add_checkpoint(Function &u, const std::string name) {
  functions_cp_.push_back(&u);
  function_names_.push_back(name);
}

//-------------------------------------------------------------------------------
void FileSaver::write_checkpoint() {
  if (functions_cp_.size() == 0 or t_save_cp_ < DOLFIN_EPS) {
    // warning("Nothing to save");
    return;
  }

  if ((time_ < eps_) || (time_ > t_save_cp_ * sample_cp_) ||
      (fabs(time_ - Time_) < eps_)) {
    set_log_active(true);
    info("Saving checkpoint to file at t = %f, sample = %d", time_, sample_cp_);
    set_log_active(false);

    for (std::size_t i = 0; i < function_names_.size(); ++i) {
      std::ostringstream ext;
      ext << sample_cp_;
      XDMFFile xdmf(folder.str() + "/checkpoint/" + function_names_[i] +
                    ext.str() + ".xdmf");
      xdmf.parameters["flush_output"] = true;
      xdmf.write_checkpoint(*functions_cp_[i], function_names_[i], time_,
                            XDMFFile::Encoding::HDF5, false);
    }

    const std::size_t mpi_rank = dolfin::MPI::rank(
        functions_cp_[0]->function_space()->mesh()->mpi_comm());
    if (mpi_rank == 0) {
      std::ofstream ffile;
      ffile.open(folder.str() + "/checkpoint/checkpoint_time.dat",
                 std::ios::app);
      ffile << time_ << "\n";
      ffile.flush();
      ffile.close();
    }
    sample_cp_++;
  }
}

//-------------------------------------------------------------------------------
void FileSaver::read_checkpoint(std::size_t counter) {
  if (functions_cp_.size() == 0) {
    warning("Nothing to read");
    return;
  }

  for (std::size_t i = 0; i < functions_cp_.size(); ++i) {
    std::ostringstream ext;
    ext << counter;
    XDMFFile xdmf(folder.str() + "/checkpoint/" + function_names_[i] +
                  ext.str() + ".xdmf");
    xdmf.parameters["flush_output"] = true;
    xdmf.read_checkpoint(*functions_cp_[i], function_names_[i]);
  }
}

//-------------------------------------------------------------------------------
void FileSaver::get_checkpoint_times(std::vector<double> &checkpoint_times) {
  std::fstream ffile;
  ffile.open(folder.str() + "/checkpoint/checkpoint_time.dat");
  if (!ffile) {
    warning("checkpoint file does not exist");
    checkpoint_times.push_back(0.0);
    return;
  }

  double t;
  while (ffile >> t) {
    checkpoint_times.push_back(t);
  }
  ffile.close();
}

///
//--- Visualization of higher order polynomials
//-------------------------------------------------------------------------------
void FileSaver::get_cg1_function(const Function &f1, Function &f2) {
  // Eval scalar function
  class Eval_Scalar : public Expression {
  public:
    Eval_Scalar(const Function &f) : Expression(), f_(f) {}
    void eval(Array<double> &values, const Array<double> &x) const {
      f_.eval(values, x);
    }

  private:
    const Function &f_;
  };

  // Eval vector function
  class Eval_Vector : public Expression {
  public:
    Eval_Vector(const Function &f)
        : Expression(f.function_space()->mesh()->topology().dim()), f_(f) {}
    void eval(Array<double> &values, const Array<double> &x) const {
      f_.eval(values, x);
    }

  private:
    const Function &f_;
  };

  // now do naive interpolation
  if (f1.value_rank() == 0) {
    Eval_Scalar eval_(f1);
    f2.interpolate(eval_);
  } else if (f1.value_rank() == 1) {
    Eval_Vector eval_(f1);
    f2.interpolate(eval_);
  } else {
    error("The functions should be either of rank 0 or rank 1 ...");
  }
}
