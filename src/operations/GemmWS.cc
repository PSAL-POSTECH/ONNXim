#include "GemmWS.h"

#include "../Model.h"

GemmWS::GemmWS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Gemm(config, model, node_proto) {}

GemmWS::GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, bool has_bias)
        : GemmWS(config, model, node_proto) {
        this->has_bias = has_bias;
}

GemmWS::GemmWS(SimulationConfig config, MappingTable& mapping_table,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> weight_shape,
               std::vector<uint32_t> output_shape)
    : Gemm(config, mapping_table, input_shape, weight_shape, output_shape) {}
void GemmWS::initialize_tiles(MappingTable& mapping_table) {
  Mapping::LoopCounts key{.N = _output_shape[_input_shape.size()-2 + Ndim] * _batch_size,
                          .C = _weight_shape[Cdim_w],
                          .M = _weight_shape[Mdim],
                          .S = 1,
                          .R = 1,
                          .Q = 1,
                          .P = 1};
  Mapping mapping;
  try {
    mapping = mapping_table.at(key);
  } catch (const std::out_of_range& e) {
    spdlog::error("Key not found: N: {} C: {} M: {} P: {} Q: {} S: {} R: {}",
      key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
  int core_id = -1; // starts from 0
  for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
    for (uint32_t M = 0; M < mapping.tile_out_loop.M; M++) {
      for (uint32_t C = 0; C < mapping.tile_out_loop.C; C++) {
        if (C == 0) {
          core_id = (core_id + 1) % _config.num_cores;
        }
        std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
          .status = Tile::Status::INITIALIZED,
          .optype = "Gemm",
          .layer_id = _id,
          .batch = N,
          .Q = 1,
          .P = 1,
          .M = M,
          .C = C,
          .S = 1,
          .R = 1,
          .accum = C != 0,
          .core_id = core_id
        });
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping);
        if (!_tiles.back().get()->instructions.size())
          _tiles.pop_back();
      }
    }
  }
}

void GemmWS::initialize_instructions(Tile* tile, Mapping mapping) {
  int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  int tout_c_offset = tile->C * mapping.tile_in_loop.C;
  int tout_n_offset = tile->batch * mapping.tile_in_loop.N;

  addr_type act_sp_base_addr = SPAD_BASE;
  addr_type weight_sp_base_addr = SPAD_BASE + mapping.tile_in_loop.N *
                                                  mapping.tile_in_loop.C *
                                                  _config.precision;


  addr_type first_addr, second_addr, third_addr, output_addr;
  first_addr = get_operand_addr(_INPUT_OPERAND);
  second_addr = get_operand_addr(_INPUT_OPERAND+1);
  third_addr = get_operand_addr(_INPUT_OPERAND+2);
  output_addr = get_operand_addr(_OUTPUT_OPERAND);

  int loop_size = _config.core_width;
  /* MOVIN BIAS */
  if (!tile->accum) {
    for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
      int M_offset = tout_m_offset + Ms;
      if (M_offset >= mapping.total_loop.N)
        break;

      int m_loop = M_offset + loop_size > mapping.total_loop.M
                      ? mapping.total_loop.M - M_offset
                      : loop_size;
      if(m_loop <= 0) break;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        std::set<addr_type> bias_addrs;
        int N_offset = tout_n_offset + Ns;
        if (N_offset >= mapping.total_loop.N)
          break;

        int n_loop = N_offset + loop_size > mapping.total_loop.N
                        ? mapping.total_loop.N - N_offset
                        : loop_size;
        for (int iter_m = 0; iter_m < m_loop; iter_m++) {
          int M = tout_m_offset + Ms + iter_m;
          if (M >= mapping.total_loop.M) continue;
          bias_addrs.insert(third_addr + make_activation_address(0, 0, 0, M, _output_shape));
        }
        if (has_bias) {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = ACCUM_SPAD_BASE +
                          (Ns * mapping.tile_in_loop.M + Ms) * _config.precision,
              .size = (uint32_t)bias_addrs.size() * n_loop,
              .src_addrs = std::vector<addr_type>(bias_addrs.begin(), bias_addrs.end()),
              .operand_id = _INPUT_OPERAND + 2}));
        } else {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::COMP,
              .dest_addr = ACCUM_SPAD_BASE +
                          (Ns * mapping.tile_in_loop.M + Ms) * _config.precision,
              .size = (uint32_t)bias_addrs.size() * n_loop,
              .src_addrs = std::vector<addr_type>(),
              .operand_id = _INPUT_OPERAND + 2}));
        }
      }
    }
  }

  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if(m_loop <= 0) break;
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += loop_size) {
      int C_offset = tout_c_offset + Cs;
      int c_loop = C_offset + loop_size > mapping.total_loop.C
                       ? mapping.total_loop.C - C_offset
                       : loop_size;
      if(c_loop <= 0) break;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                         ? mapping.total_loop.N - N_offset
                         : loop_size;
        if(n_loop <= 0) break;
        addr_type act_sp_addr =
            act_sp_base_addr +
            (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type weight_sp_addr =
            weight_sp_base_addr +
            (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        /* MOVIN Activation */
        if (Ms == 0) {
          std::set<addr_type> input_set;
          for (int iter_c = 0; iter_c < c_loop; iter_c++) {
            for (int iter_n = 0; iter_n < n_loop; iter_n++) {
              int N = N_offset + iter_n;
              int C = C_offset + iter_c;
              input_set.insert(
                  first_addr + make_activation_address(N, 0, 0, C, _input_shape));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .size = (uint32_t)input_set.size(),
              .src_addrs =
                  std::vector<addr_type>(input_set.begin(), input_set.end()),
              .operand_id = _INPUT_OPERAND,
              .tile_k = mapping.tile_in_loop.C,
              .tile_n = mapping.tile_in_loop.N}));
        }
        /* MOVIN Weight */
        if (Ns == 0) {
          std::set<addr_type> weight_set;
          for (int iter_m = 0; iter_m < m_loop; iter_m++) {
            for (int iter_c = 0; iter_c < c_loop; iter_c++) {
              int M = M_offset + iter_m;
              int C = C_offset + iter_c;
              std::vector<uint32_t> weight_shape_4d;
              weight_shape_4d.resize(4);
              weight_shape_4d[Mdim] = _weight_shape[0];
              weight_shape_4d[Sdim] = 1;
              weight_shape_4d[Rdim] = 1;
              weight_shape_4d[Cdim_w] = _weight_shape[1];
              weight_set.insert(
                  second_addr + make_weight_address(0, 0, M, C, weight_shape_4d));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = weight_sp_addr,
              .size = (uint32_t)weight_set.size(),
              .src_addrs =
                  std::vector<addr_type>(weight_set.begin(), weight_set.end()),
              .operand_id = _INPUT_OPERAND + 1,
              .tile_m = mapping.tile_in_loop.M,
              .tile_k = mapping.tile_in_loop.C}));
        }
        std::set<addr_type> output_set;
        for (int iter_n = 0; iter_n < n_loop; iter_n++) {
          for (int iter_m = 0; iter_m < m_loop; iter_m++) {
            int N = N_offset + iter_n;
            int M = M_offset + iter_m;
            output_set.insert(output_addr + make_activation_address(N, 0, 0, M, _output_shape));
          }
        }

        /*Compute */
        if (Ns == 0) {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::GEMM_PRELOAD,
              .dest_addr = out_sp_addr,
              // Accumulat buffer already allocated
              .size = (uint32_t)output_set.size(),
              .compute_size = (uint32_t)n_loop,
              .src_addrs =
                  std::vector<addr_type>{act_sp_addr, weight_sp_addr}}));
        } else {
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::GEMM,
              .dest_addr = out_sp_addr,
              // Accumulat buffer already allocated
              .size = (uint32_t)output_set.size(),
              .compute_size = (uint32_t)n_loop,
              .src_addrs =
                  std::vector<addr_type>{act_sp_addr, weight_sp_addr}}));
        }
        /*MOVOUT result at the last loop*/
        if (Cs == mapping.tile_in_loop.C - loop_size && Ms == mapping.tile_in_loop.M - loop_size){
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVOUT,
              .dest_addr = out_sp_addr,
              .size = (uint32_t)output_set.size(),
              .src_addrs =
                  std::vector<addr_type>(output_set.begin(), output_set.end()),
              .operand_id = _OUTPUT_OPERAND}));
        }
      }
    }
  }
}