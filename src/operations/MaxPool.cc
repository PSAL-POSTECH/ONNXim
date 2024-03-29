#include "MaxPool.h"

#include <robin_hood.h>

#include "../Model.h"
#include "../Tensor.h"

MaxPool::MaxPool(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
  int kernel_dim = 0;
  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "kernel_shape") {
      spdlog::trace("kernel_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _kernel_shape.push_back(attribute.ints(i));
      }
      kernel_dim = attribute.ints_size();
    } else if (attribute.name() == "strides") {
      for (int i = 0; i < attribute.ints_size(); i++) {
        _strides.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "auto_pad") {
    } else if (attribute.name() == "pads") {
      spdlog::trace("padn_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pads.push_back(attribute.ints(i));
      }
    }
  }

  /* We assume conv2d */
  assert(kernel_dim == 2);
  std::vector<uint32_t> output_shape;
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  output_shape.resize(input_shape.size());
  output_shape[Ndim] = input_shape[Ndim];
  output_shape[Cdim] = input_shape[Cdim];
  for (int i = 0; i < kernel_dim; i++) {
    output_shape[Hdim + i] =
        (uint32_t)ceil(((float)input_shape[Hdim + i] + _pads[i] +
                        _pads[i + 2] - (_kernel_shape[i] - 1)) /
                       (float)_strides[i]);
  }

  spdlog::trace("output name : {} {}", node_proto.output(0).c_str(),
                output_shape);
  Tensor* predefined_tensor = _model->find_tensor(node_proto.output(0));
  if (predefined_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), output_shape, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    predefined_tensor->redefine_tensor(_id, output_shape);
  }

  _tiles.push_back(
      Tile{.status = Tile::Status::INITIALIZED, .layer_id = _id, .batch = 0, .skip = true});
}

MaxPool::MaxPool(const MaxPool& src) : Operation(src) {
  _kernel_shape = src._kernel_shape;
  _strides = src._strides;
  _dilations = src._dilations;
  _pads = src._pads;
}

/*TODO: implement this */
void MaxPool::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {} ", _name);
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();  

  uint32_t h_shift = (input_shape[Hdim] - _kernel_shape[0]) / _strides[0] + 1;
  uint32_t w_shift = (input_shape[Wdim] - _kernel_shape[1]) / _strides[1] + 1;

  for (uint32_t N = 0; N < input_shape[Ndim]; N++) {
    for (uint32_t C = 0; C < input_shape[Cdim]; C++) {
      for (uint32_t H = 0; H < h_shift; H++) {
        for (uint32_t W = 0; W < w_shift; W++) {
          _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                                .optype = "MaxPool",
                                .layer_id = _id,
                                .batch = N,
                                .Q = H,
                                .P = W,
                                .C = C,
                                .skip = true});
          initialize_instructions(_tiles.back(), Mapping{});
        }
      } 
    }
  }
}

void MaxPool::initialize_instructions(Tile& tile, Mapping mapping) {
  // std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  // std::vector<uint32_t> input_shape = get_input(0)->get_dims();

  // uint32_t h_kernel = _kernel_shape[0];
  // uint32_t w_kernel = _kernel_shape[1];
  // uint32_t kernel_size = h_kernel * w_kernel;   // Compare size
  // uint32_t compare_size_in_vector = _config.vector_process_bit /       // Maximum compare size in a vector unit
  //                                     (_config.precision * 8);

  // uint32_t N = tile.batch;
  // uint32_t C = tile.C;
  // uint32_t tout_q_offset = tile.Q * _strides[0];
  // uint32_t tout_p_offset = tile.P * _strides[1];

  // uint32_t total_compare = 0;   // How many COMP need to pull 1 max element
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
  //     input_set.insert(make_activation_address(N, q_offset, p_offset, 
  //                           C, input_shape));
  //   }
  // }

  // std::string input_id = fmt::format("INPUT-{}-{}-{}-{}-{}", tile.layer_id,
  //                                    N, tout_q_offset, tout_p_offset, C);

  // /* MOVIN elements to COMP */
  // tile.instructions.push_back(
  //       Instruction{.opcode = Opcode::MOVIN,
  //                   .id = input_id,
  //                   .addrs = std::vector<addr_type>(
  //                            input_set.begin(), input_set.end())});

  // std::set<addr_type> output_set;
  // std::string output_id = fmt::format("OUT-{}-{}-{}-{}-{}", tile.layer_id,
  //                                     N, tout_q_offset, tout_p_offset, C);
  
  // output_set.insert(make_activation_address(N, tout_q_offset, 
  //                                           tout_p_offset, C, output_shape));

  // for (int i=0; i<total_compare; i++)
  //   tile.instructions.push_back(
  //         Instruction{.opcode = Opcode::COMP,
  //                     .tile_size = compare_size_in_vector,
  //                     .id = fmt::format("COMP-{}-{}-{}-{}-{}", tile.layer_id, N,
  //                                       tout_q_offset, tout_p_offset, C),
  //                     .dependent_ids = std::vector<std::string>{input_id},
  //                     .dest_id = output_id});

  // tile.instructions.push_back(
  //       Instruction{.opcode = Opcode::MOVOUT,
  //                   .id = output_id,
  //                   .addrs = std::vector<addr_type>(output_set.begin(), 
  //                                           output_set.end())});  
}