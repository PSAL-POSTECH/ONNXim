#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "TileGraph.h"
#include "operations/OperationFactory.h"

TileGraph::TileGraph(std::string onnx_path, SimulationConfig config) {
  std::ifstream model_istream(onnx_path);
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  _model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  _root_node_id = generate_id();
  _config = config;

  auto input = _model_proto.graph().input();
  for (auto node_proto : _model_proto.graph().node()) {
    for (auto attribute : node_proto.attribute()) {
      if (attribute.name() == "torchsim_start") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _start.push_back(attribute.ints(i));
      } else if (attribute.name() == "torchsim_end") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _end.push_back(attribute.ints(i));
      } else if (attribute.name() == "torchsim_stride") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _stride.push_back(attribute.ints(i));
      } else if (attribute.name() == "torchsim_base_addr") {
        _base_addr_ptr = attribute.s();
      } else if (attribute.name() == "torchsim_element_size") {
        _precision = attribute.i();
      } else if (attribute.name() == "torchsim_stride_list") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _stride_list.push_back(attribute.ints(i));
      } else if (attribute.name() == "torchsim_tile_size") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _tile_size.push_back(attribute.ints(i));
      } else if (attribute.name() == "torchsim_tile_stride") {
        for (int i = 0; i < attribute.ints_size(); i++)
          _tile_stride.push_back(attribute.ints(i));
      } else if (attribute.name() == "inst0") {
        // printf("inst0: %s\n", attribute.s().c_str()); // TODO: is this really necessary?
      } else if (attribute.name() == "torchsim_cycle") {
        _cycle = attribute.i();
      }
    }
    initialize_tile(node_proto.op_type());
  }
  _tile_generate();
}

void TileGraph::initialize_tile(std::string op_type) {
  if (op_type == "load_node") {
    addr_type dest = _config.align_address(SPAD_BASE + _base_addr);
    uint32_t memory_req_size = (_tile_size[0] * _tile_size[1] * _precision - 1) / _config.dram_req_size + 1;
    _instructions.push_back(
      Instruction{.opcode = Opcode::MOVIN,
                  .dest_addr = dest,
                  .src_addr = _base_addr,
                  .size = memory_req_size,
                  .base_addr = 0});
    _src_addrs.push_back(dest);
    _base_addr_update();
  } else if (op_type == "compute_node") {
    _instructions.push_back(
      Instruction{.opcode = Opcode::COMP,
                  .compute_cycle = _cycle,
                  .dest_addr = _base_addr,
                  .size = 1,
                  .src_addrs = _src_addrs});
  } else if (op_type == "store_node") {
    uint32_t memory_req_size = (_tile_size[0] * _tile_size[1] * _precision - 1) / _config.dram_req_size + 1;
    _instructions.push_back(
      Instruction{.opcode = Opcode::MOVOUT,
                  .dest_addr = _base_addr,
                  .src_addr = _base_addr,
                  .size = memory_req_size,
                  .base_addr = SPAD_BASE});
    _base_addr_update();
  }
}

// make several tiles from tile graph infos
void TileGraph::_tile_generate() {
  _tile_index_generate();
  for (auto index : _tile_index) {
    _tiles.push_back(
      Tile{.status = Tile::Status::INITIALIZED,
            .optype = "example",
            .layer_id = _root_node_id,
            .batch = 0,
            .Q = 0, // TODO: remove legacy information
            .P = 0,
            .M = 0,
            .C = 0,
            .S = _tile_size[0],
            .R = _tile_size[1]});
    uint32_t offset = index * _precision;
    for (auto inst : _instructions) {
      Instruction new_inst = inst; // copy instruction
      for (auto &addr : new_inst.src_addrs)
        addr = addr + offset;
      new_inst.src_addr = new_inst.src_addr + offset;
      new_inst.dest_addr = new_inst.dest_addr + offset;
      _tiles.back().instructions.push_back(new_inst);
    }
  }
}

void TileGraph::_base_addr_update() {
  uint64_t tensor_size = 1;
  for (int i = 0; i < _start.size(); i++) {
    tensor_size *= (_end[i] - _start[i]);
  }
  _base_addr += (tensor_size * _precision);
  if (_base_addr_map.find(_base_addr_ptr) == _base_addr_map.end())
    _base_addr_map[_base_addr_ptr] = _base_addr;
  else
    _base_addr = _base_addr_map[_base_addr_ptr];
}

void TileGraph::_tile_index_generate() {
  for (int i = _start[0]; i < _end[0]; i += _stride[0]) // initialize inner most loop
    _tile_index.push_back(i);
  uint32_t loop_size = 0;
  for (int i = 1; i < _start.size(); i++) { // make tile index from inner to outer
    loop_size += _end[i - 1] - _start[i - 1];
    std::vector<uint32_t> temp;
    for (int j = _start[i]; j < _end[i]; j++) {
      for (auto k : _tile_index)
        temp.push_back(j * loop_size + k);
    }
    _tile_index = temp;
  }
}