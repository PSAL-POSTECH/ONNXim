/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "CommandLineParser.h"

namespace po = boost::program_options;

void CommandLineParser::parse(int argc, char** argv) noexcept(false) {
  po::store(po::parse_command_line(argc, argv, options_description),
            variables_map);
  po::notify(variables_map);
}

void CommandLineParser::print_help_message_if_required() const noexcept {
  if (variables_map.count("help") > 0) {
    std::cout << options_description << std::endl;
    exit(0);
  }
}
