#include <fstream>

#include "Simulator.h"
#include "helper/CommandLineParser.h"
#include "operations/OperationFactory.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "model", "Path for the onnx file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose one_model or two_model");

  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  std::string model_base_path = "./models";
  cmd_parser.set_if_defined("model", &model_base_path);
  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  cmd_parser.set_if_defined("config", &config_path);

  json config_json;
  std::ifstream config_file(config_path);
  config_file >> config_json;
  config_file.close();
  SimulationConfig config = initialize_config(config_json);
  OperationFactory::initialize(config);

  std::string models_list_path;
  cmd_parser.set_if_defined("models_list", &models_list_path);
  std::ifstream models_list_file(models_list_path);
  json models_list;
  models_list_file >> models_list;
  models_list_file.close();
  auto simulator = std::make_unique<Simulator>(config);
  for (json model_config : models_list["models"]) {
    std::string model_name = model_config["name"];
    std::string input_name = model_config["input_name"];
    std::string onnx_path =
        fmt::format("{}/{}.onnx", model_base_path, model_name);
    auto tile_graph = std::make_unique<TileGraph>(onnx_path, config);
    spdlog::info("Launching tiles\n");
    simulator->run_tile(std::move(tile_graph));
  }

  return 0;
}
