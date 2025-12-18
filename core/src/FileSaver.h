// Copyright (C) 2015 Murtazo Nazarov
// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef __FILE_SAVER_H
#define __FILE_SAVER_H

#include <dolfin/function/Function.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/io/File.h>
#include <dolfin/io/XDMFFile.h>

///
namespace dolfin
{
}

///
using dolfin::Function;
using dolfin::FunctionSpace;
using dolfin::Parameters;
using dolfin::File;
using dolfin::XDMFFile;

namespace fem_varden_flow
{
class FileSaver
{
public:

  ///
  FileSaver(double const & time, Parameters const & parameters);

  ///
  FileSaver();

  ///
  ~FileSaver();

  ///
  void init();

  ///
  void update();

  ///
  void add(const Function& u, const std::string& name);

  ///
  void add(const Function& u, const Function& u_fine, const std::string& name);

  ///
  void add(const Function& u, std::shared_ptr<FunctionSpace> V_CG1_fine,
	   const std::string& name);

  ///
  void save();

  ///
  std::ostringstream folder;

  ///
  void add_checkpoint(Function & u, std::string name);

  ///
  void write_checkpoint();

  ///
  void read_checkpoint(std::size_t counter);

  ///
  void get_checkpoint_times(std::vector<double> & checkpoint_times);


  ///
  double t_save()
  {
    return t_save_;
  }

  ///
  int sample()
  {
    return sample_;
  }

private:

  ///
  double const & time_;
  Parameters const& parameters_;
  double Time_;
  double t_save_;
  double t_save_cp_;
  int sample_;
  int sample_cp_;
  double hmin_;
  std::string s_file_type_="vtk";
  double const eps_ = 1.e-10;
  
  ///
  std::vector<const Function*> functions_;
  std::vector<std::pair<const Function*,
    std::shared_ptr<const FunctionSpace> >> functions_P_;
  std::vector<const Function*> functions_fine_;
  std::vector<File *> files_fine_;
  std::vector<XDMFFile *> xfiles_fine_;

  ///
  File * file_;
  XDMFFile * xfile_;
  std::vector<File *> files_;
  std::vector<XDMFFile *> xfiles_;

  ///
  XDMFFile * xfile_cp_;
  std::vector<XDMFFile *> xfiles_cp_;
  std::vector<std::string> function_names_;
  std::vector<Function*> functions_cp_;

  ///
  //std::size_t sample_cp_;

  ///
  void get_cg1_function(const Function& f1, Function& f2);

};
}

#endif
