#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "TileGraph.h"
#include "operations/OperationFactory.h"

TileGraph::TileGraph(std::string onnx_path) {
  std::ifstream model_istream(onnx_path);
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  _model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  _root_node_id = generate_id();

  auto input = _model_proto.graph().input();
  for(auto node_proto : _model_proto.graph().node()) {
    for(auto attribute : node_proto.attribute()) {
      if(attribute.name() == "torchsim_base_addr") {
        _base_addr = attribute.i();
      }
      if(attribute.name() == "torchsim_element_size") {
        _precision = attribute.i();
      }
      if(attribute.name() == "torchsim_stride_list") {
        for(int i = 0; i < attribute.ints_size(); i++) {
          _stride_list.push_back(attribute.ints(i));
        }
      }
      if(attribute.name() == "torchsim_tile_size") {
        for(int i = 0; i < attribute.ints_size(); i++) {
          _tile_size.push_back(attribute.ints(i));
        }
      }
      if(attribute.name() == "torchsim_tile_stride") {
        for(int i = 0; i < attribute.ints_size(); i++) {
          _tile_stride.push_back(attribute.ints(i));
        }
      }
      if(attribute.name() == "inst0") {
        // printf("inst0: %s\n", attribute.s().c_str()); // TODO: is this really necessary?
      }
      if(attribute.name() == "torchsim_cycle") {
        _cycle = attribute.i();
      }
    }
    initialize_tile(node_proto.op_type());
  }
  tile_generate();
}

void TileGraph::initialize_tile(std::string op_type) {
  if (op_type == "load_node") {
    _instructions.push_back(
      Instruction{.opcode = Opcode::MOVIN,
                  .size = _tile_size[0],
                  .base_addr = _base_addr});
  } else if (op_type == "compute_node") {
    _instructions.push_back(
      Instruction{.opcode = Opcode::COMP,
                  .size = _tile_size[0],
                  .base_addr = _base_addr});
  } else if (op_type == "store_node") {
    _instructions.push_back(
      Instruction{.opcode = Opcode::MOVOUT,
                  .size = _tile_size[0],
                  .base_addr = _base_addr});
  }
}

// make several tiles from tile graph infos
void TileGraph::tile_generate() {
  for (int id = 0; id < _tile_stride[0]; id ++) { // FIXME: loop range would be different
    _tiles.push_back(
      Tile{.status = Tile::Status::INITIALIZED,
            .optype = "example",
            .batch = 0,
            .Q = 0,
            .P = 0,
            .M = 0,
            .C = 0,
            .S = _tile_size[0],
            .R = _tile_size[1]});
    for (int i = 0; i < _instructions.size(); i++) {
      addr_type addr = _instructions[i].base_addr + id * (_tile_size[0] * _tile_size[1]) * _precision; // TODO: calculate address
      _instructions[i].src_addrs = std::vector<addr_type>(addr); // TODO: src address could be more than 1
      _tiles.back().instructions.push_back(_instructions[i]);
    }
  }
}