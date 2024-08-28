#include "Conv.h"

#include <robin_hood.h>

#include "../Model.h"
#include "../Tensor.h"

Conv::Conv(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  int kernel_dim = 0;
  _activation_fused = false;
  _pool_fused = false;
  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "kernel_shape") {
      spdlog::trace("kernel_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _kernel_shape.push_back(attribute.ints(i));
      }
      kernel_dim = attribute.ints_size();
    } else if (attribute.name() == "strides") {
      spdlog::trace("stride_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _strides.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "auto_pad") {
    } else if (attribute.name() == "dilations") {
      spdlog::trace("dilation_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _dilations.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "pads") {
      spdlog::trace("padn_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pads.push_back(attribute.ints(i));  // left, right, top, down
      }
    } else if (attribute.name() == "activation") {
      spdlog::trace("Activatoion: {}", attribute.s().c_str());
      _activation_type = attribute.s();
      _activation_fused = true;
    } else if (attribute.name() == "group") {
      spdlog::trace("Group {}", attribute.i());
      _group = attribute.i();
    } else if (attribute.name() == "pool") {
      _pool_fused = true;
      _pool_type = attribute.s();
    } else if (attribute.name() == "pool_kernel_shape") {
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pool_kernel_shape.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "pool_strides") {
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pool_strides.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "pool_pads") {
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pool_pads.push_back(attribute.ints(i));
      }
    }
  }
  _skip_connection_fused = false;

  /* We assume conv2d */
  assert(kernel_dim == 2);

  _input_shape = get_input(0)->get_dims();
  int shape_offset = 0;
  _conv_out_shape.resize(4);
  _conv_out_shape[Cdim] = get_input(1)->get_dims()[Mdim];  // CoCiHW
  _conv_out_shape[Ndim] = _input_shape[Ndim];
  for (int i = 0; i < kernel_dim; i++) {
    _conv_out_shape[Hdim + i] =
        (uint32_t)((float)_input_shape[Hdim + i] + _pads[i] + _pads[i + 2] -
                   (_dilations[i] * (_kernel_shape[i] - 1)) - 1) /
            (float)_strides[i] +
        1;
  }
  _weight_shape = get_input(1)->get_dims();
  std::vector<uint32_t> output_shape;
  if (_pool_fused) {
    _pool_out_shape.resize(4);
    _pool_out_shape[Cdim] = _conv_out_shape[Cdim];
    _pool_out_shape[Ndim] = _conv_out_shape[Ndim];
    for (int i = 0; i < kernel_dim; i++) {
      _pool_out_shape[Hdim + i] =
          (uint32_t)((float)_conv_out_shape[Hdim + i] + _pool_pads[i] +
                     _pool_pads[i + kernel_dim] - _pool_kernel_shape[i]) /
              (float)_pool_strides[i] +
          1;
    }
    output_shape = _pool_out_shape;
  } else {
    output_shape = _conv_out_shape;
  }

  spdlog::trace("output_shape : {}", _conv_out_shape);
  if (_inputs.size() == 3) {
    /* TODO: Handle this part */
    spdlog::trace("BN fused");
  }
  if (_inputs.size() == 4) {
    /* TODO: Handle this part */
    spdlog::trace("BN fused");
    spdlog::trace("Skip_connection fused");
    for (int i = 0; i < 4; i++) {
      assert(get_input(3)->get_dims()[i] == output_shape[i]);
    }
    _skip_connection_fused = true;
  }

  Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
  if (pre_defind_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), output_shape, _config.precision, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    pre_defind_tensor->redefine_tensor(_id, output_shape);
  }
}

Conv::Conv(const Conv& src) : Operation(src) {
  _kernel_shape = src._kernel_shape;
  _strides = src._strides;
  _dilations = src._dilations;
  _pads = src._pads;
  _activation_fused = src._activation_fused;
  _activation_type = src._activation_type;
  _bathnorm_fused = src._bathnorm_fused;
  _skip_connection_fused = src._skip_connection_fused;
}

void Conv::im2col_nhwc() {
  std::vector<uint32_t> output_shape = _conv_out_shape;
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  std::vector<uint32_t> weight_shape = get_input(1)->get_dims();

  int height_col = output_shape[Hdim];
  int width_col = output_shape[Wdim];
  int dkernel_h = _dilations[0] * (_kernel_shape[0] - 1) + 1;
  int dkernel_w = _dilations[1] * (_kernel_shape[1] - 1) + 1;
  int stride_h = _strides[0];
  int stride_w = _strides[1];
  int kernel_h = _kernel_shape[Rdim];
  int kernel_w = _kernel_shape[Sdim];
  int channels = input_shape[Cdim];
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
      .status = Tile::Status::INITIALIZED, .optype = "im2col", .layer_id = _id});
  for (int h = 0; h < height_col; h++) {
    int h_pad = -_pads[2] + h * stride_h;
    addr_type data_col_tmp =
        h * width_col * kernel_h * kernel_w * channels * _config.precision;
    int w_pad = -_pads[0];
    for (int w = 0; w < width_col; w++) {
      int r = 0;
      std::set<addr_type> src_set;
      std::set<addr_type> dest_set;
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += _dilations[0], ++r) {
        int s = 0;
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += _dilations[1], ++s) {
          if (ih >= 0 && ih < input_shape[Hdim] && iw >= 0 &&
              iw < input_shape[Wdim]) {
            for (int g = 0; g < _group; g++) {
              addr_type dest_addr = _config.align_address(
                  data_col_tmp + ((g * kernel_h + r) * kernel_w + s) *
                                     (channels / _group) * _config.precision);
              addr_type src_addr = _config.align_address(
                  (ih * input_shape[Wdim] + iw) * channels * _config.precision +
                  g * (channels / _group) * _config.precision);
              dest_set.insert(dest_addr);
              src_set.insert(src_addr);
            }
          } else {
            for (int g = 0; g < _group; g++) {
              for (int i = 0; i < channels / _group; i++) {
                addr_type dest_addr = _config.align_address(
                    data_col_tmp +
                    ((g * kernel_h + r) * kernel_w + s) * (channels / _group) *
                        _config.precision +
                    i * _config.precision);
                dest_set.insert(dest_addr);
              }
            }
          }
        }
      }

      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = SPAD_BASE,
          .size = (uint32_t)dest_set.size(),
          .src_addrs = std::vector<addr_type>(src_set.begin(), src_set.end()),
          .operand_id = _INPUT_OPERAND}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                      .opcode = Opcode::IM2COL,
                      .dest_addr = SPAD_BASE,
                      .size = (uint32_t)dest_set.size()}));
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVOUT,
          .dest_addr = SPAD_BASE,
          .size = (uint32_t)dest_set.size(),
          .src_addrs = std::vector<addr_type>(dest_set.begin(), dest_set.end()),
          .operand_id = _INPUT_OPERAND + 4}));

      data_col_tmp += kernel_h * kernel_w * channels;
      w_pad += _strides[1];
    }
  }
  _tiles.push_back(std::move(tile));
  _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR}));
}

Conv::Conv(SimulationConfig config, MappingTable& mapping_table,
               convInfo info, uint32_t target_core)
    : Operation(config, mapping_table, target_core){
  _kernel_shape = info.kernel_shape;
  _strides = info.strides;
  _dilations = info.dilations;
  _pads = info.pads;
  _input_shape = info.input_shape;
  _weight_shape = info.weight_shape;
  _conv_out_shape = info.conv_out_shape;
  _pool_out_shape = info.pool_out_shape;
  _activation_fused = info.activation_fused;
  _group = info.group;
  _activation_type = info.activation_type;
  _bathnorm_fused = info.bathnorm_fused;
  _skip_connection_fused = info.skip_connection_fused;
  _pool_fused = info.pool_fused;
  _pool_type = info.pool_type;
  _pool_kernel_shape = info.pool_kernel_shape;
  _pool_strides = info.pool_strides;
  _pool_pads =info.pool_pads;
}