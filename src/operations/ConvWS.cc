#include "ConvWS.h"

#include "../Model.h"
#include "../Tensor.h"

ConvWS::ConvWS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Conv(config, model, node_proto) {}

ConvWS::ConvWS(const Conv& src) : Conv(src) {}

ConvWS::ConvWS(SimulationConfig config, MappingTable& mapping_table, convInfo info)
    : Conv(config, mapping_table, info) {}
/* TODO: handle depthwise convolutoin (Important) */
/* TODO: handle grouped convolutoin (less important) */
void ConvWS::initialize_tiles(MappingTable& mapping_table) {
  int tile_h_size = _config.core_height;
  int tile_w_size = _config.core_width;
  int precision = _config.precision;
  spdlog::trace("initialize_tile {} ", _name);
  std::vector<uint32_t> output_shape = _conv_out_shape;
  /*Im2Col + Matrix multiplicaiton for Group convoution*/
  if (_group != 1) {
    im2col_nhwc();
    for (uint32_t group = 0; group < _group; group++) {
      _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                            .optype = "Matmul",
                            .layer_id = _id,
                            .batch = 0,
                            .Q = 0,
                            .P = 0,
                            .M = _weight_shape[Mdim] / _group * group,
                            .C = _weight_shape[Cdim_w] / _group * group,
                            .S = 0,
                            .R = 0,
                            .accum = false});
      initialize_matmul_instructions(_tiles.back());
      spdlog::info("Group convolution {}", _id);
    }
    return;
  }

  /* Implicit im2col for Conventional convolution */
  Mapping::LoopCounts key{.N = output_shape[Ndim],
                          .C = _weight_shape[Cdim_w],
                          .M = _weight_shape[Mdim],
                          .S = _weight_shape[Sdim],
                          .R = _weight_shape[Rdim],
                          .Q = output_shape[Hdim],
                          .P = output_shape[Wdim]};
  Mapping mapping;
  try {
    mapping = mapping_table.at(key);
  } catch (const std::out_of_range& e) {
    spdlog::error("Key not found: N: {} C: {} M: {} P: {} Q: {} S: {} R: {}",
      key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
  assert(mapping.tile_in_loop.C > 1);
  if (_pool_fused) {
    assert(mapping.tile_in_loop.P >= _pool_kernel_shape[0] &&
           mapping.tile_in_loop.P % _pool_kernel_shape[0] == 0);
    assert(mapping.tile_in_loop.Q >= _pool_kernel_shape[1] &&
           mapping.tile_in_loop.Q % _pool_kernel_shape[1] == 0);
  }
  /* TODO: Spatially seperate polocy for weight stationary */
  for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
    for (uint32_t P = 0; P < mapping.tile_out_loop.P; P++) {
      for (uint32_t Q = 0; Q < mapping.tile_out_loop.Q; Q++) {
        for (uint32_t M = 0; M < mapping.tile_out_loop.M; M++) {
          for (uint32_t S = 0; S < mapping.tile_out_loop.S; S++) {
            for (uint32_t R = 0; R < mapping.tile_out_loop.R; R++) {
              for (uint32_t C = 0; C < mapping.tile_out_loop.C; C++) {
                _tiles.push_back(
                    Tile{.status = Tile::Status::INITIALIZED,
                         .optype = "Conv",
                         .layer_id = _id,
                         .batch = N,
                         .Q = Q,
                         .P = P,
                         .M = M,
                         .C = C,
                         .S = S,
                         .R = R,
                         .accum = (C != 0 || R != 0 ||
                                   S != 0)}); /* Accum input channel data*/
                initialize_instructions(_tiles.back(), mapping);
              }
            }
          }
        }
      }
    }
  }
  assert(_tiles.size() > 0);
}

void ConvWS::initialize_instructions(Tile& tile, Mapping mapping) {
  std::vector<uint32_t> output_shape = _conv_out_shape;
  int sram_allocation = 0;
  int act_allocation = 0;
  int tout_n_offset = tile.batch * mapping.tile_in_loop.N;
  int tout_m_offset = tile.M * mapping.tile_in_loop.M;
  int tout_q_offset = tile.Q * mapping.tile_in_loop.Q;
  int tout_p_offset = tile.P * mapping.tile_in_loop.P;
  int tout_c_offset = tile.C * mapping.tile_in_loop.C;
  int tout_s_offset = tile.S * mapping.tile_in_loop.S;
  int tout_r_offset = tile.R * mapping.tile_in_loop.R;
  int input_h_size = (mapping.tile_in_loop.Q - 1) * _strides[0] +
                     _dilations[0] * (_kernel_shape[0] - 1) + 1;
  int input_w_size = (mapping.tile_in_loop.P - 1) * _strides[1] +
                     _dilations[1] * (_kernel_shape[1] - 1) + 1;
  int input_h_offset =
      tout_q_offset - _dilations[0] * (_kernel_shape[0] - 1) / 2;
  int input_w_offset =
      tout_p_offset - _dilations[1] * (_kernel_shape[1] - 1) / 2;
  addr_type act_sp_base_addr = SPAD_BASE;
  addr_type weight_sp_base_addr =
      SPAD_BASE + mapping.tile_in_loop.N * input_h_size * input_w_size *
                      mapping.tile_in_loop.C * _config.precision;

  int loop_size = _config.core_width;
  robin_hood::unordered_map<std::string, Instruction> inst_map;

  /*MOVIN Bias*/
  
  // if (_bathnorm_fused && tile.C == 0 && tile.S == 0 && tile.R == 0) {
  //   for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
  //     int N = tout_n_offset + Ns;
  //     for (int Qs = 0; Qs < mapping.tile_in_loop.Q; Qs++) {
  //       int Q = tout_q_offset + Qs;
  //       for (int Ps = 0; Ps < mapping.tile_in_loop.P; Ps += loop_size) {
  //         for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
  //           int p_loop = tout_p_offset + Ps + loop_size > mapping.total_loop.P
  //                            ? mapping.total_loop.P - tout_p_offset - Ps
  //                            : loop_size;
  //           int m_loop = tout_m_offset + Ms + loop_size > mapping.total_loop.M
  //                            ? mapping.total_loop.M - tout_m_offset - Ms
  //                            : loop_size;
  //           if(m_loop <= 0) break;
  //           addr_type out_sp_addr =
  //               ACCUM_SPAD_BASE +
  //               make_activation_address(
  //                   Ns, Qs, Ps, Ms,
  //                   std::vector<uint32_t>{
  //                       mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
  //                       mapping.tile_in_loop.P, mapping.tile_in_loop.M});

  //           std::set<addr_type> bias_dram_addrs;

  //           for (int M_iter = 0; M_iter < m_loop; M_iter++) {
  //             int M = tout_m_offset + Ms + M_iter;
  //             bias_dram_addrs.insert(
  //                 make_activation_address(0, 0, 0, M, output_shape));
  //           } 
  //           tile.instructions.push_back(
  //               Instruction{.opcode = Opcode::MOVIN,
  //                           .dest_addr = out_sp_addr,
  //                           .size = (uint32_t)bias_dram_addrs.size() * p_loop,
  //                           .src_addrs = std::vector<addr_type>(
  //                               bias_dram_addrs.begin(), bias_dram_addrs.end()),
  //                           .operand_id = _INPUT_OPERAND + 2});
  //         }
  //       }
  //     }
  //   }
  // }

  // /*MOVIN Skip-connection*/
  // if (_skip_connection_fused && tile.C == 0 && tile.S == 0 && tile.R == 0) {
  //   for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
  //     int N = tout_n_offset + Ns;
  //     for (int Qs = 0; Qs < mapping.tile_in_loop.Q; Qs++) {
  //       int Q = tout_q_offset + Qs;
  //       for (int Ps = 0; Ps < mapping.tile_in_loop.P; Ps += loop_size) {
  //         for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
  //           int p_loop = tout_p_offset + Ps + loop_size > mapping.total_loop.P
  //                            ? mapping.total_loop.P - tout_p_offset - Ps
  //                            : loop_size;
  //           int m_loop = tout_m_offset + Ms + loop_size > mapping.total_loop.M
  //                            ? mapping.total_loop.M - tout_m_offset - Ms
  //                            : loop_size;
  //           if(m_loop <= 0) break;
  //           addr_type out_sp_addr =
  //               ACCUM_SPAD_BASE +
  //               make_activation_address(
  //                   Ns, Qs, Ps, Ms,
  //                   std::vector<uint32_t>{
  //                       mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
  //                       mapping.tile_in_loop.P, mapping.tile_in_loop.M});
  //           std::set<addr_type> out_dram_addrs;
  //           for (int P_iter = 0; P_iter < p_loop; P_iter++) {
  //             int P = tout_p_offset + Ps + P_iter;
  //             for (int M_iter = 0; M_iter < m_loop; M_iter++) {
  //               int M = tout_m_offset + Ms + M_iter;
  //               out_dram_addrs.insert(
  //                   make_activation_address(N, Q, P, M, output_shape));
  //             }
  //           }
  //           tile.instructions.push_back(
  //               Instruction{.opcode = Opcode::MOVIN,
  //                           .dest_addr = out_sp_addr,
  //                           .size = (uint32_t)out_dram_addrs.size(),
  //                           .src_addrs = std::vector<addr_type>(
  //                               out_dram_addrs.begin(), out_dram_addrs.end()),
  //                           .operand_id = _INPUT_OPERAND + 3});
  //         }
  //       }
  //     }
  //   }
  // }

  /* MOVIN Activation data */
  std::set<addr_type> act_addr_set;

  for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
    for (int Hs = 0; Hs < input_h_size; Hs++) {
      for (int Ws = 0; Ws < input_w_size; Ws++) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs++) {
          int N = tout_n_offset + Ns;
          int H = input_h_offset + Hs;
          int W = input_w_offset + Ws;
          int C = tout_c_offset + Cs;
          if(C  >= mapping.total_loop.C) break;
          if (H < 0 || H >= _input_shape[Hdim] || W < 0 ||
              W >= _input_shape[Wdim])
            continue;
          act_addr_set.insert(make_activation_address(N, H, W, C, _input_shape));
        }
      }
    }
  }

  tile.instructions.push_back(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = act_sp_base_addr,
      .size = (uint32_t)act_addr_set.size(),
      .src_addrs =
          std::vector<addr_type>(act_addr_set.begin(), act_addr_set.end()),
      .operand_id = _INPUT_OPERAND});
  sram_allocation += act_addr_set.size();
  act_allocation += act_addr_set.size();
  /* MOVIN Weight data */
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int m_loop = tout_m_offset + Ms + loop_size > mapping.total_loop.M 
                     ? mapping.total_loop.M - tout_m_offset - Ms
                     : loop_size;
    if(m_loop <= 0) break;
    for (int Ss = 0; Ss < mapping.tile_in_loop.S; Ss++) {
      for (int Rs = 0; Rs < mapping.tile_in_loop.R; Rs++) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += loop_size) {
          int c_loop = tout_c_offset + Cs + loop_size > mapping.total_loop.C
                           ? mapping.total_loop.C - tout_c_offset - Cs
                           : loop_size;
          if(c_loop <= 0) break;
          c_loop = Cs + c_loop > mapping.tile_in_loop.C ? mapping.tile_in_loop.C - Cs : c_loop;
          addr_type weight_sp_addr =
              weight_sp_base_addr +
              make_weight_address(
                  Ss, Rs, Ms, Cs,
                  std::vector<uint32_t>{
                      mapping.tile_in_loop.M, mapping.tile_in_loop.C,
                      mapping.tile_in_loop.S, mapping.tile_in_loop.R});
          std::set<addr_type> weight_set;
          int m_offset = tout_m_offset + Ms;
          int s_offset = tout_s_offset + Ss;
          int r_offset = tout_r_offset + Rs;
          int c_offset = tout_c_offset + Cs;

          for (int m_iter = 0; m_iter < m_loop; m_iter++) {
            for (int c_iter = 0; c_iter < c_loop; c_iter++) {
              int M = m_offset + m_iter;
              int C = c_offset + c_iter;
              weight_set.insert(
                  make_weight_address(s_offset, r_offset, M, C, _weight_shape));
            }
          }
          tile.instructions.push_back(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = weight_sp_addr,
              .size = (uint32_t)weight_set.size(),
              .src_addrs =
                  std::vector<addr_type>(weight_set.begin(), weight_set.end()),
              .operand_id = _INPUT_OPERAND + 1});
          sram_allocation += weight_set.size();
        }
      }
    }
  }

  /* Compute */
  int q_loop_size = 1;
  int p_loop_size = loop_size;
  if (_pool_fused) {
    q_loop_size = _pool_kernel_shape[0];
    p_loop_size = _pool_kernel_shape[1];
  }
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    if(tout_m_offset + Ms >= mapping.total_loop.M) break;
    for (int Ss = 0; Ss < mapping.tile_in_loop.S; Ss += 1) {
      for (int Rs = 0; Rs < mapping.tile_in_loop.R; Rs += 1) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs += loop_size) {
          if(tout_c_offset + Cs >= mapping.total_loop.C) break;
          for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
            for (int Qs = 0; Qs < mapping.tile_in_loop.Q; Qs += q_loop_size) {
              if(tout_q_offset + Qs >= mapping.total_loop.Q) break;
              for (int Ps = 0; Ps < mapping.tile_in_loop.P; Ps += p_loop_size) {
                if(tout_p_offset + Ps >= mapping.total_loop.P) break;
                addr_type weight_sp_addr =
                    weight_sp_base_addr +
                    make_weight_address(
                        Ss, Rs, Ms, Cs,
                        std::vector<uint32_t>{
                            mapping.tile_in_loop.M, mapping.tile_in_loop.C,
                            mapping.tile_in_loop.S, mapping.tile_in_loop.R});
                addr_type out_sp_addr =
                    ACCUM_SPAD_BASE +
                    make_activation_address(
                        Ns, Qs, Ps, Ms,
                        std::vector<uint32_t>{
                            mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
                            mapping.tile_in_loop.P, mapping.tile_in_loop.M});
                int p_loop = Ps + p_loop_size > mapping.tile_in_loop.P
                                 ? mapping.tile_in_loop.P - Ps
                                 : p_loop_size;
                p_loop = tout_p_offset + Ps + p_loop > mapping.total_loop.P
                                ? mapping.total_loop.P - Ps - tout_p_offset
                                : p_loop;
                int compute_size = p_loop * q_loop_size;
                if (Ns == 0 && Qs == 0 && Ps == 0) {
                  tile.instructions.push_back(
                      Instruction{.opcode = Opcode::GEMM_PRELOAD,
                                  .dest_addr = out_sp_addr,
                                  .size = (uint32_t)compute_size * _config.precision / _config.dram_req_size,
                                  .compute_size = /*Todo*/ (uint32_t)compute_size,
                                  .src_addrs = std::vector<addr_type>{
                                      act_sp_base_addr, weight_sp_addr}});
                } else {
                  tile.instructions.push_back(
                      Instruction{.opcode = Opcode::GEMM,
                                  .dest_addr = out_sp_addr,
                                  .size = (uint32_t)compute_size * _config.precision / _config.dram_req_size,
                                  .compute_size = /*Todo*/ (uint32_t)compute_size,
                                  .src_addrs = std::vector<addr_type>{
                                      act_sp_base_addr, weight_sp_addr}});
                }
              }
            }
          }
        }
      }
    }
  }

  /* MOVOUT at last iteration */
  if (tile.C == mapping.tile_out_loop.C - 1 &&
      tile.R == mapping.tile_out_loop.R - 1 &&
      tile.S == mapping.tile_out_loop.S - 1) {
    /* Pooling not fused */
    for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
      int N = tout_n_offset + Ns;
      for (int Qs = 0; Qs < mapping.tile_in_loop.Q; Qs += q_loop_size) {
        if(tout_q_offset + Qs >= mapping.total_loop.Q) break;
        for (int Ps = 0; Ps < mapping.tile_in_loop.P; Ps += p_loop_size) {
          for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
            int p_loop = tout_p_offset + Ps + p_loop_size > mapping.total_loop.P
                             ? mapping.total_loop.P - tout_p_offset - Ps
                             : p_loop_size;
            int m_loop = tout_m_offset + Ms + loop_size > mapping.total_loop.M
                             ? mapping.total_loop.M - tout_m_offset - Ms
                             : loop_size;
            if(m_loop <= 0) break;
            if(p_loop <= 0) break;
            int q_loop = q_loop_size;
            addr_type out_sp_addr =
                ACCUM_SPAD_BASE +
                make_activation_address(
                    Ns, Qs, Ps, Ms,
                    std::vector<uint32_t>{
                        mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
                        mapping.tile_in_loop.P, mapping.tile_in_loop.M});
            std::set<addr_type> out_dram_addrs;
            if (_pool_fused) {
              q_loop = q_loop / _pool_kernel_shape[0];
              p_loop = p_loop / _pool_kernel_shape[1];
            }
            for (int Q_iter = 0; Q_iter < q_loop; Q_iter++) {
              int Q = tout_q_offset + Qs + Q_iter;
              for (int P_iter = 0; P_iter < p_loop; P_iter++) {
                int P = tout_p_offset + Ps + P_iter;
                for (int M_iter = 0; M_iter < m_loop; M_iter++) {
                  int M = tout_m_offset + Ms + M_iter;
                  if (_pool_fused)
                    out_dram_addrs.insert(
                        make_activation_address(N, Q, P, M, _pool_out_shape));
                  else
                    out_dram_addrs.insert(
                        make_activation_address(N, Q, P, M, _conv_out_shape));
                }
              }
            }
            if (_pool_fused) {
              tile.instructions.push_back(
                  Instruction{.opcode = Opcode::MOVOUT_POOL,
                              .dest_addr = out_sp_addr,
                              .size = (uint32_t)out_dram_addrs.size(),
                              .src_addrs = std::vector<addr_type>(
                                  out_dram_addrs.begin(), out_dram_addrs.end()),
                              .operand_id = _OUTPUT_OPERAND});
            } else {
              tile.instructions.push_back(
                  Instruction{.opcode = Opcode::MOVOUT,
                              .dest_addr = out_sp_addr,
                              .size = (uint32_t)out_dram_addrs.size(),
                              .src_addrs = std::vector<addr_type>(
                                  out_dram_addrs.begin(), out_dram_addrs.end()),
                              .operand_id = _OUTPUT_OPERAND});
            }
          }
        }
      }
    }
  }
  spdlog::trace("Layer {} Sram allocation size {} act {} weight {}", _name,
                sram_allocation, act_allocation,
                sram_allocation - act_allocation);
  assert(sram_allocation <= _config.spad_size KB / _config.dram_req_size / 2);
  assert(act_allocation <= _config.spad_size KB / _config.dram_req_size / 2);
}

void ConvWS::initialize_matmul_instructions(Tile& tile) {
  std::vector<uint32_t> output_shape = _conv_out_shape;
  uint32_t kernel_size = _weight_shape[Sdim] * _weight_shape[Rdim];
  uint32_t channels = _input_shape[Cdim];
  uint32_t N_dim_size =
      output_shape[Ndim] * output_shape[Hdim] * output_shape[Wdim];
  uint32_t C_dim_size = kernel_size * _weight_shape[Cdim_w] / _group;
  int loop_size = _config.core_width;
  addr_type weight_offset =
      tile.M * kernel_size * (channels / _group) * _config.precision;

  addr_type act_sp_base = SPAD_BASE;
  addr_type weight_sp_base =
      SPAD_BASE + C_dim_size * N_dim_size * _config.precision;
  addr_type out_sp_base = ACCUM_SPAD_BASE;

  /*Bias */
  for (int M = tile.M; M < tile.M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile.M + (_weight_shape[Mdim] / _group)
                     ? tile.M + (_weight_shape[Mdim] / _group) - M
                     : loop_size;
    for (int N = 0; N < N_dim_size; N += loop_size) {
      int n_loop = N + loop_size > N_dim_size ? N_dim_size - N : loop_size;
      addr_type bias_sp_addr =
          out_sp_base +
          (N * _weight_shape[Mdim] / _group + M) * _config.precision;
      std::set<addr_type> skip_addrs;
      for (int m_iter = 0; m_iter < m_loop; m_iter++) {
        skip_addrs.insert(
            _config.align_address((M + m_iter) * _config.precision));
      }
      tile.instructions.push_back(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = bias_sp_addr,
          .size = (uint32_t)skip_addrs.size() * n_loop,
          .src_addrs =
              std::vector<addr_type>(skip_addrs.begin(), skip_addrs.end()),
          .operand_id = _INPUT_OPERAND + 2});
    }
  }

  /*Skip */
  for (int M = tile.M; M < tile.M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile.M + (_weight_shape[Mdim] / _group)
                     ? tile.M + (_weight_shape[Mdim] / _group) - M
                     : loop_size;
    for (int N = 0; N < N_dim_size; N += loop_size) {
      int n_loop = N + loop_size > N_dim_size ? N_dim_size - N : loop_size;
      addr_type skip_sp_addr =
          out_sp_base +
          (N * _weight_shape[Mdim] / _group + M) * _config.precision;
      std::set<addr_type> skip_addrs;
      for (int n_iter = 0; n_iter < n_loop; n_iter++) {
        for (int m_iter = 0; m_iter < m_loop; m_iter++) {
          skip_addrs.insert(_config.align_address(
              ((N + n_iter) * output_shape[Cdim] + M + m_iter) *
              _config.precision));
        }
      }
      tile.instructions.push_back(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = skip_sp_addr,
          .size = (uint32_t)skip_addrs.size(),
          .src_addrs =
              std::vector<addr_type>(skip_addrs.begin(), skip_addrs.end()),
          .operand_id = _INPUT_OPERAND + 3});
    }
  }

  for (int M = tile.M; M < tile.M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile.M + (_weight_shape[Mdim] / _group)
                     ? tile.M + (_weight_shape[Mdim] / _group) - M
                     : loop_size;
    for (int C = 0; C < C_dim_size; C += loop_size) {
      int c_loop = C + loop_size > C_dim_size ? C_dim_size - C : loop_size;
      for (int N = 0; N < N_dim_size; N += loop_size) {
        int n_loop = N + loop_size > N_dim_size ? N_dim_size - N : loop_size;
        addr_type act_sp_addr =
            act_sp_base + (N * C_dim_size + C) * _config.precision;
        addr_type weight_sp_addr =
            weight_sp_base + (M * C_dim_size + C) * _config.precision;
        addr_type out_sp_addr =
            out_sp_addr +
            (N * _weight_shape[Mdim] / _group + M) * _config.precision;
        /*MOVIN activation*/
        if (M == tile.M) {
          std::set<addr_type> act_addr;
          for (int c_iter = 0; c_iter < c_loop; c_iter++) {
            for (int n_iter = 0; n_iter < n_loop; n_iter++) {
              act_addr.insert(make_activation_address(
                  N + n_iter, 0, 0, C + c_iter,
                  std::vector<uint32_t>{N_dim_size, 0, 0,
                                        C_dim_size * _group}));
            }
          }
          tile.instructions.push_back(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .size = (uint32_t)act_addr.size(),
              .src_addrs =
                  std::vector<addr_type>(act_addr.begin(), act_addr.end()),
              .operand_id = _INPUT_OPERAND});
        }

        /* MOVIN weight */
        if (N == 0) {
          std::set<addr_type> weight_addr;
          for (int m_iter = 0; m_iter < m_loop; m_iter++) {
            for (int c_iter = 0; c_iter < c_loop; c_iter++) {
              weight_addr.insert(make_weight_address(
                  0, 0, M + m_iter, C + c_iter,
                  std::vector<uint32_t>{_weight_shape[Mdim], C_dim_size * _group,
                                        0, 0}));
            }
          }
          tile.instructions.push_back(
              Instruction{.opcode = Opcode::MOVIN,
                          .dest_addr = weight_sp_addr,
                          .size = (uint32_t)weight_addr.size(),
                          .src_addrs = std::vector<addr_type>(
                              weight_addr.begin(), weight_addr.end()),
                          .operand_id = _INPUT_OPERAND + 1});
        }

        /*MOVOUT */
        if (C + loop_size >= C_dim_size) {
          std::set<addr_type> out_addrs;
          for (int n_iter = 0; n_iter < n_loop; n_iter++) {
            for (int m_iter = 0; m_iter < m_loop; m_iter++) {
              out_addrs.insert(_config.align_address(
                  ((N + n_iter) * output_shape[Cdim] + M + m_iter) *
                  _config.precision));
            }
          }
          tile.instructions.push_back(Instruction{
              .opcode = Opcode::MOVOUT,
              .dest_addr = out_sp_addr,
              .size = (uint32_t)out_addrs.size(),
              .src_addrs =
                  std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
              .operand_id = _OUTPUT_OPERAND});
        }
      }
    }
  }
}