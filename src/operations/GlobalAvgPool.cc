#include "GlobalAvgPool.h"
#include "../Tensor.h"
#include "../Model.h"


GlobalAvgPool::GlobalAvgPool(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core) 
  : Operation(config, model, node_proto, target_core) {

  /* We assume conv2d */
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  std::vector<uint32_t> output_shape = input_shape;
  output_shape[Hdim] = output_shape[Wdim] = 1;

  /* Kernel H W size are same as input H W */
  _kernel_shape.push_back(input_shape[Hdim]);
  _kernel_shape.push_back(input_shape[Wdim]);
  _strides = _kernel_shape;

  // _kernel_shape[0] = _strides[0] = input_shape[Hdim];
  // _kernel_shape[1] = _strides[1] = input_shape[Wdim];

  spdlog::trace("output name {}", node_proto.output(0).c_str());
  Tensor* predefined_tensor = _model->find_tensor(node_proto.output(0));
  if(predefined_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(_id, node_proto.output(0), output_shape,
      _config.precision, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  }
  else {
    predefined_tensor->redefine_tensor(_id, output_shape);
  }  
}


GlobalAvgPool::GlobalAvgPool(const GlobalAvgPool& src) 
  : Operation(src) {
  _kernel_shape = src._kernel_shape;
  _strides = src._strides;
} 

/* TODO: Implement this */
void GlobalAvgPool::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {}", _name);
  std::vector<uint32_t> output_shape = get_output(0)->get_dims();

  _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                        .optype = "GlobalAvgPool",
                        .layer_id = _id,
                        .batch = 1,
                        .Q = 0,
                        .P = 0,
                        .C = 1,
                        .skip = true}));
  initialize_instructions(_tiles.back().get(), Mapping{});
}

void GlobalAvgPool::initialize_instructions(Tile* tile, Mapping mapping) {
  // std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  // std::vector<uint32_t> input_shape = get_input(0)->get_dims();

  // uint32_t h_kernel = _kernel_shape[0];
  // uint32_t w_kernel = _kernel_shape[1];
  // uint32_t kernel_size = h_kernel * w_kernel;
  // uint32_t compare_size_in_vector = _config.vector_process_bit /
  //                                     (_config.precision * 8);

  // uint32_t N = tile->batch;
  // uint32_t C = tile->C;
  // uint32_t tout_q_offset = tile->Q * _strides[0];
  // uint32_t tout_p_offset = tile->P * _strides[1];

  // uint32_t total_compare = 0;
  // uint32_t tmp = kernel_size;

  // while (tmp > compare_size_in_vector) {
  //   int quotient = tmp / compare_size_in_vector;
  //   int remainder = tmp % compare_size_in_vector;

  //   total_compare += quotient;
  //   tmp = quotient + remainder;
  // }
  // total_compare += 1;

  // std::set<addr_type> input_set;

  // for (int q_offset = 0; q_offset < h_kernel; q_offset++) {
  //   for (int p_offset = 0; p_offset < w_kernel; p_offset++) {
  //     input_set.insert(make_activation_address(N, q_offset, 
  //                           p_offset, C, input_shape));
  //   }
  // }

  // std::string input_id = fmt::format("INPUT-{}-{}-{}-{}-{}", tile->layer_id,
  //                                    N, tout_q_offset, tout_p_offset, C);

  // tile->instructions.push_back(
  //       Instruction{.opcode = Opcode::MOVIN,
  //                   .id = input_id,
  //                   .addrs = std::vector<addr_type>(
  //                            input_set.begin(), input_set.end())});

  // std::set<addr_type> output_set;
  // std::string output_id = fmt::format("OUT-{}-{}-{}-{}-{}", tile->layer_id,
  //                                     N, tout_q_offset, tout_p_offset, C);
  
  // output_set.insert(make_activation_address(N, tout_q_offset, 
  //                                           tout_p_offset, C, output_shape));

  // for (int i=0; i<total_compare; i++)
  //   tile->instructions.push_back(
  //         Instruction{.opcode = Opcode::COMP,
  //                     .tile_size = compare_size_in_vector,
  //                     .id = fmt::format("COMP-{}-{}-{}-{}-{}", tile->layer_id, N,
  //                                       tout_q_offset, tout_p_offset, C),
  //                     .dependent_ids = std::vector<std::string>{input_id},
  //                     .dest_id = output_id});

  // tile->instructions.push_back(
  //       Instruction{.opcode = Opcode::MOVOUT,
  //                   .id = output_id,
  //                   .addrs = std::vector<addr_type>(output_set.begin(), 
  //                                           output_set.end())});
}