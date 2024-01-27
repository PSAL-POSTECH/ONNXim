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
}

void TileGraph::initialize_tile(std::string op_type) {
  uint64_t addr = _base_addr; // TODO: calculate address
  if (op_type == "load_node") {
    instructions.push_back(
      Instruction{.opcode = Opcode::MOVIN,
                  .size = _tile_size[0],
                  .base_addr = addr});
  } else if (op_type == "compute_node") {
    instructions.push_back(
      Instruction{.opcode = Opcode::COMP,
                  .size = _tile_size[0],
                  .base_addr = addr});
  } else if (op_type == "store_node") {
    instructions.push_back(
      Instruction{.opcode = Opcode::MOVOUT,
                  .size = _tile_size[0],
                  .base_addr = addr});
  }
}