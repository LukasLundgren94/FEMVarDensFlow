// Copyright (C) 2025 Lukas Lundgren
// Licensed under the MIT License – see the LICENSE file for details

#ifndef FEM_VARDEN_FLOW_CONFIGURABLEPARAMETERS_H
#define FEM_VARDEN_FLOW_CONFIGURABLEPARAMETERS_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/variant.hpp>
#include <dolfin/parameter/Parameters.h>

namespace fem_varden_flow
{
class ConfigurableParameters
{
public:
  enum class ParameterType
  {
    Int,
    Double,
    String,
    Bool
  };

  using ParameterValue = boost::variant<int, double, bool, std::string>;

  struct ParameterSpec
  {
    std::string name;
    ParameterType type;
    ParameterValue default_value;
    std::string description;
    std::string qualified_name;
    bool alias_unique;
  };

  struct ParameterCategory
  {
    std::string name;
    std::string description;
    std::vector<ParameterSpec> parameters;
  };

  explicit ConfigurableParameters(const std::string& config_path = std::string());

  ~ConfigurableParameters() = default;


  dolfin::Parameters to_dolfin_parameters(const std::string& name = "configurable_parameters") const;

  const std::map<std::string, ParameterValue>& values() const
  {
    return values_;
  }

  static const std::vector<ParameterCategory>& categories();
  static const std::vector<ParameterSpec>& specification();

  static dolfin::Parameters default_category_parameters(const std::string& category_name);

  static void write_default_to(const std::string& file_path);

private:
  void load_defaults();
  void load_from_file(const std::string& file_path);
  void load_from_json(const std::string& file_path);
  void load_from_legacy_format(const std::string& file_path);

  void apply_leaf(const std::string& key, const std::string& raw_value);
  void set_value(const std::string& key, const ParameterValue& value);

  ParameterValue parse_with_type(const std::string& key, ParameterType type, const std::string& raw_value) const;

  void report_defaults_used(const std::string& source) const;
  const ParameterSpec* find_spec(const std::string& key) const;

  std::map<std::string, ParameterValue> values_;
  std::set<std::string> overridden_parameters_;
  std::set<std::string> unknown_parameters_;
  std::string current_source_;
};
}

#endif
