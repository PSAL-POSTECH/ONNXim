#include "Mapping.h"

#include <fstream>
#include <sstream>
#include <string>
// #include "Common.h"

MappingTable::MappingTable (SimulationConfig config) {
  _mapping_table = _MappingTable();
  _config = config;
}

MappingTable MappingTable::parse_mapping_file(
    std::string mapping_path, SimulationConfig config) {
  MappingTable map = MappingTable(config);
  std::ifstream mapping_file;
  std::string line;
  mapping_file.open(mapping_path);
  if (mapping_file.fail()) {
    spdlog::error("Invalid mapping file path : {}", mapping_path);
    throw std::runtime_error("Data error");
  }
  while (getline(mapping_file, line)) {
    Mapping mapping(line);
    map[mapping.total_loop] = mapping;
    spdlog::trace("N {} C {} M {} S {} R {} Q {} P {}", 
      mapping.total_loop.N,
      mapping.total_loop.C,
      mapping.total_loop.M,
      mapping.total_loop.S,
      mapping.total_loop.R,
      mapping.total_loop.Q,
      mapping.total_loop.P
      );
  }
  mapping_file.close();
  return map;
}

const Mapping& MappingTable::fallback_mapping(Mapping::LoopCounts &key) {
  return _mapping_table[key];
}

const Mapping& MappingTable::at(Mapping::LoopCounts &key) {
  auto it = _mapping_table.find(key);
  if (it != _mapping_table.end())
    return it->second;
  else
    return fallback_mapping(key);
}

uint32_t Mapping::LoopCounts::get_loop(Mapping::LoopName name) {
  switch (name) {
    case Mapping::LoopName::N:
      return N;
    case Mapping::LoopName::C:
      return C;
    case Mapping::LoopName::M:
      return M;
    case Mapping::LoopName::S:
      return S;
    case Mapping::LoopName::R:
      return R;
    case Mapping::LoopName::P:
      return P;
    case Mapping::LoopName::Q:
      return Q;
    default:
      assert(0);
      /* Unreachable */
      return 0;
  }
}

Mapping::Mapping(std::string mapping_line) {
  std::string total_tile;
  std::string out_tile;
  std::string in_tile;
  std::string loop_elem;
  std::stringstream level_parse;
  std::stringstream loop_parse;
  spatial_M = 1;
  spatial_P = 1;
  spatial_Q = 1;
  spatial_C = 1;
  spatial_R = 1;
  spatial_S = 1;
  level_parse << mapping_line;

  // Parse Total Loop
  getline(level_parse, total_tile, '-');
  loop_parse << total_tile;
  getline(loop_parse, loop_elem, ' ');
  while (getline(loop_parse, loop_elem, ' ')) {
    LoopName loop_name;
    int loop_count = std::stoi(loop_elem.substr(1));
    switch (loop_elem.at(0)) {
      case 'N':
        total_loop.N = loop_count;
        break;
      case 'C':
        total_loop.C = loop_count;
        break;
      case 'M':
        total_loop.M = loop_count;
        break;
      case 'S':
        total_loop.S = loop_count;
        break;
      case 'R':
        total_loop.R = loop_count;
        break;
      case 'Q':
        total_loop.Q = loop_count;
        break;
      case 'P':
        total_loop.P = loop_count;
        break;
      default:
        assert(0);
    }
  }

  // Parse Outer Loop
  getline(level_parse, out_tile, '-');
  loop_parse.clear();
  loop_parse << out_tile;
  getline(loop_parse, loop_elem, ' ');
  getline(loop_parse, loop_elem, ' ');
  while (getline(loop_parse, loop_elem, ' ')) {
    LoopName loop_name;
    int loop_count = std::stoi(loop_elem.substr(1));
    switch (loop_elem.at(0)) {
      case 'N':
        loop_name = LoopName::N;
        tile_out_loop.N = loop_count;
        break;
      case 'C':
        loop_name = LoopName::C;
        tile_out_loop.C = loop_count;
        break;
      case 'M':
        loop_name = LoopName::M;
        tile_out_loop.M = loop_count;
        break;
      case 'S':
        loop_name = LoopName::S;
        tile_out_loop.S = loop_count;
        break;
      case 'R':
        loop_name = LoopName::R;
        tile_out_loop.R = loop_count;
        break;
      case 'Q':
        loop_name = LoopName::Q;
        tile_out_loop.Q = loop_count;
        break;
      case 'P':
        loop_name = LoopName::P;
        tile_out_loop.P = loop_count;
        break;
      default:
        assert(0);
    }
    tile_out_loop_order.push_back(loop_name);
  }

  // Parse L4
  
  while (getline(level_parse, in_tile, '-')) {
    loop_parse.clear();
    loop_parse << in_tile;
    getline(loop_parse, loop_elem, ' ');
    getline(loop_parse, loop_elem, ' ');
    while (getline(loop_parse, loop_elem, ' ')) {
      LoopName loop_name;
      int loop_count;
      switch (loop_elem.at(0)) {
        case 'N':
          tile_in_loop.N *= std::stoi(loop_elem.substr(1));
          total_loop.N *= std::stoi(loop_elem.substr(1));
          break;
        case 'C':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_C = loop_count;
          } 
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.C *= loop_count;
          break;
        case 'M':
          if (loop_elem.find('X') != std::string::npos) {
            size_t pos = loop_elem.find('X');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_M = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.M *= loop_count;
          break;
        case 'S':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_S = loop_count;
          } 
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.S *= loop_count;
          break;
        case 'R':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_R = loop_count;
          } 
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.R *= loop_count;
          break;
        case 'Q':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_Q = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.Q *= loop_count;
          break;
        case 'P':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_P = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.P *= loop_count;
          break;
        default:
          assert(0);
      }
    }
  }
}
