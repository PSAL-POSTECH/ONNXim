#include "ConvWS.h"

#include "../Model.h"
#include "../Tensor.h"

ConvWS::ConvWS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Conv(config, model, node_proto, target_core) {}

ConvWS::ConvWS(const Conv& src) : Conv(src) {}

ConvWS::ConvWS(SimulationConfig config, MappingTable& mapping_table, convInfo info)
    : Conv(config, mapping_table, info) {}
/* TODO: handle depthwise convolutoin (Important) */
/* TODO: handle grouped convolutoin (less important) */
void ConvWS::initialize_tiles(MappingTable& mapping_table) {
  int tile_h_size = _config.core_config[target_core].core_height;
  int tile_w_size = _config.core_config[target_core].core_width;
  int precision = _config.precision;
  spdlog::trace("initialize_tile {} ", _name);
  std::vector<uint32_t> output_shape = _conv_out_shape;
  /*Im2Col + Matrix multiplicaiton for Group convoution*/
  if (_group != 1) {
    im2col_nhwc();
    for (uint32_t group = 0; group < _group; group++) {
      std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
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
      _tiles.push_back(std::move(tile));
      initialize_matmul_instructions(_tiles.back().get());
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
                          .P = output_shape[Wdim],
                          .Padding = _pads.at(0),
                          .Stride = _strides.at(0),
                          .target_core = target_core};
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
  int core_id = -1; // starts from 0
  for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
    for (uint32_t P = 0; P < mapping.tile_out_loop.P; P++) {
      for (uint32_t Q = 0; Q < mapping.tile_out_loop.Q; Q++) {
        for (uint32_t M = 0; M < mapping.tile_out_loop.M; M++) {
          for (uint32_t S = 0; S < mapping.tile_out_loop.S; S++) {
            for (uint32_t R = 0; R < mapping.tile_out_loop.R; R++) {
              for (uint32_t C = 0; C < mapping.tile_out_loop.C; C++) {
                if (C == 0 && R == 0 && S == 0) {
                  core_id = (core_id + 1) % _config.num_cores;
                }
                std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile {
                  .status = Tile::Status::INITIALIZED,
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
                            S != 0),
                  .core_id = core_id
                });
                //spdlog::info("Outer P: {}, Q:{}", P, Q);
                _tiles.push_back(std::move(tile)); /* Accum input channel data*/
                initialize_instructions(_tiles.back().get(), mapping);
                if (!_tiles.back()->instructions.size())
                  _tiles.pop_back();
              }
            }
          }
        }
      }
    }
  }
  assert(_tiles.size() > 0);
}

void ConvWS::initialize_instructions(Tile* tile, Mapping mapping) {
  std::vector<uint32_t> output_shape = _conv_out_shape;
  int sram_allocation = 0;
  int act_allocation = 0;
  int tout_n_offset = tile->batch * mapping.tile_in_loop.N;
  int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  int tout_q_offset = tile->Q * mapping.tile_in_loop.Q;
  int tout_p_offset = tile->P * mapping.tile_in_loop.P;
  int tout_c_offset = tile->C * mapping.tile_in_loop.C;
  int tout_s_offset = tile->S * mapping.tile_in_loop.S;
  int tout_r_offset = tile->R * mapping.tile_in_loop.R;
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

  if (tout_n_offset >= mapping.total_loop.N ||
      tout_m_offset >= mapping.total_loop.M ||
      tout_q_offset >= mapping.total_loop.Q ||
      tout_p_offset >= mapping.total_loop.P ||
      tout_c_offset >= mapping.total_loop.C ||
      tout_s_offset >= mapping.total_loop.S ||
      tout_r_offset >= mapping.total_loop.R) {
    return;
  }

  int loop_size = _config.core_config[target_core].core_width;
  robin_hood::unordered_map<std::string, Instruction> inst_map;

  /*MOVIN Bias*/
  
  // if (_bathnorm_fused && tile->C == 0 && tile->S == 0 && tile->R == 0) {
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
  //           tile->instructions.push_back(
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
  // if (_skip_connection_fused && tile->C == 0 && tile->S == 0 && tile->R == 0) {
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
  //           tile->instructions.push_back(
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

  addr_type first_addr;
  addr_type second_addr;
  addr_type output_addr;

  first_addr = get_operand_addr(_INPUT_OPERAND);
  second_addr = get_operand_addr(_INPUT_OPERAND+1);
  output_addr = get_operand_addr(_OUTPUT_OPERAND);
  for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
    for (int Hs = 0; Hs < input_h_size; Hs+=_strides[0]) {
      for (int Ws = 0; Ws < input_w_size; Ws+=_strides[1]) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs++) {
          int N = tout_n_offset + Ns;
          int H = input_h_offset + Hs;
          int W = input_w_offset + Ws;
          int C = tout_c_offset + Cs;
          if(C  >= mapping.total_loop.C) break;
          if (H < 0 || H >= _input_shape[Hdim] || W < 0 ||
              W >= _input_shape[Wdim])
            continue;
          act_addr_set.insert(first_addr + make_activation_address(N, H, W, C, _input_shape));
        }
      }
    }
  }

  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVIN,
      .dest_addr = act_sp_base_addr,
      .size = (uint32_t)act_addr_set.size(),
      .src_addrs =
          std::vector<addr_type>(act_addr_set.begin(), act_addr_set.end()),
      .operand_id = _INPUT_OPERAND}));
  sram_allocation += act_addr_set.size();
  act_allocation += act_addr_set.size();
  /* MOVIN Weight data */
  int tile_idx=0;
  std::set<addr_type> sp_addr_set;
  for (int Ms = 0; Ms < mapping.tile_in_loop.M && Ms + tout_m_offset < mapping.total_loop.M; Ms += loop_size) {
    /* Boundary check */
    int m_loop = std::min(mapping.total_loop.M - tout_m_offset - Ms, static_cast<unsigned int>(loop_size));
    m_loop = std::min(static_cast<int>(mapping.tile_in_loop.M) - Ms, m_loop);
    if(m_loop <= 0) break;
    for (int Ss = 0; Ss < mapping.tile_in_loop.S; Ss++) {
      for (int Rs = 0; Rs < mapping.tile_in_loop.R; Rs++) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C && Cs + tout_c_offset < mapping.total_loop.C; Cs += loop_size, tile_idx++) {
          /* Boundary check */
          int c_loop = std::min(mapping.total_loop.C - tout_c_offset - Cs, static_cast<unsigned int>(loop_size));
          c_loop = std::min(static_cast<int>(mapping.tile_in_loop.C) - Cs, c_loop);
          if(c_loop <= 0) break;

          addr_type weight_sp_addr = \
              weight_sp_base_addr + loop_size * loop_size * \
              (tile_idx * (mapping.tile_in_loop.S * mapping.tile_in_loop.R) + Ss * mapping.tile_in_loop.R + Rs);
          weight_sp_addr = _config.align_address(weight_sp_addr);

          if (sp_addr_set.find(weight_sp_addr) == sp_addr_set.end())
            sp_addr_set.insert(weight_sp_addr);
          else
            spdlog::info("error! weight spad address is invalid!");
          std::set<addr_type> weight_set;
          int m_offset = tout_m_offset + Ms;
          int s_offset = tout_s_offset + Ss;
          int r_offset = tout_r_offset + Rs;
          int c_offset = tout_c_offset + Cs;

          for (int m_iter = 0; m_iter < m_loop; m_iter++) {
            for (int c_iter = 0; c_iter < c_loop; c_iter++) {
              int M = m_offset + m_iter;
              int C = c_offset + c_iter;
              weight_set.insert( second_addr + \
                  make_weight_address(s_offset, r_offset, M, C, _weight_shape));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = weight_sp_addr,
              .size = (uint32_t)weight_set.size(),
              .src_addrs =
                  std::vector<addr_type>(weight_set.begin(), weight_set.end()),
              .operand_id = _INPUT_OPERAND + 1}));
          sram_allocation += weight_set.size();
        }
      }
    }
  }
  /* Compute */
  int q_loop_size = loop_size;
  int p_loop_size = loop_size;
  if (_pool_fused) {
    q_loop_size = _pool_kernel_shape[0];
    p_loop_size = _pool_kernel_shape[1];
  }
  tile_idx=0;
  for (int Ms = 0; Ms < mapping.tile_in_loop.M && Ms + tout_m_offset < mapping.total_loop.M; Ms += loop_size) {
    int m_loop = std::min(mapping.total_loop.M - tout_m_offset - Ms, static_cast<unsigned int>(loop_size));
    m_loop = std::min(static_cast<int>(mapping.tile_in_loop.M) - Ms, m_loop);
    if(m_loop <= 0) break;
    for (int Ss = 0; Ss < mapping.tile_in_loop.S; Ss++) {
      for (int Rs = 0; Rs < mapping.tile_in_loop.R; Rs++) {
        for (int Cs = 0; Cs < mapping.tile_in_loop.C && Cs + tout_c_offset < mapping.total_loop.C; Cs += loop_size, tile_idx++) {
          int c_loop = std::min(mapping.total_loop.C - tout_c_offset - Cs, static_cast<unsigned int>(loop_size));
          c_loop = std::min(static_cast<int>(mapping.tile_in_loop.C) - Cs, c_loop);
          if(c_loop <= 0) break;

          addr_type weight_sp_addr = \
              weight_sp_base_addr + loop_size * loop_size * \
              (tile_idx * (mapping.tile_in_loop.S * mapping.tile_in_loop.R) + Ss * mapping.tile_in_loop.R + Rs);
          weight_sp_addr = _config.align_address(weight_sp_addr);

          for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
            for (int Qs = 0; Qs < mapping.tile_in_loop.Q && Qs + tout_q_offset < mapping.total_loop.Q; Qs += q_loop_size) {
              int q_loop = std::min(mapping.total_loop.Q - tout_q_offset - Qs, static_cast<unsigned int>(q_loop_size));
              q_loop = std::min(static_cast<int>(mapping.tile_in_loop.Q) - Qs, q_loop);
              if(q_loop <= 0) break;
              for (int Ps = 0; Ps < mapping.tile_in_loop.P && Ps + tout_p_offset < mapping.total_loop.P; Ps += p_loop_size) {
                int p_loop = std::min(mapping.total_loop.P - tout_p_offset - Ps, static_cast<unsigned int>(p_loop_size));
                p_loop = std::min(static_cast<int>(mapping.tile_in_loop.P) - Ps, p_loop);
                if(p_loop <= 0) break;

                addr_type out_sp_addr =
                    ACCUM_SPAD_BASE +
                    make_activation_address(
                        Ns, Qs, Ps, Ms,
                        std::vector<uint32_t>{
                            mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
                            mapping.tile_in_loop.P, mapping.tile_in_loop.M});
                out_sp_addr = _config.align_address(out_sp_addr);

                int compute_size = p_loop * q_loop;
                if (Ns == 0 && Qs == 0 && Ps == 0) {
                  tile->instructions.push_back(std::make_unique<Instruction>(
                      Instruction{.opcode = Opcode::GEMM_PRELOAD,
                                  .dest_addr = out_sp_addr,
                                  .size = (uint32_t)compute_size * _config.precision / _config.dram_req_size,
                                  .compute_size = /*Todo*/ (uint32_t)compute_size,
                                  .src_addrs = std::vector<addr_type>{
                                      act_sp_base_addr, weight_sp_addr},
                                  .tile_m = static_cast<unsigned int>(m_loop),
                                  .tile_k = static_cast<unsigned int>(c_loop),
                                  .tile_n = static_cast<unsigned int>(compute_size)
                                  }));
                } else {
                  //spdlog::info("GEMM m: {}, c: {}, compute: {}", m_loop, c_loop, compute_size);
                  tile->instructions.push_back(std::make_unique<Instruction>(
                      Instruction{.opcode = Opcode::GEMM,
                                  .dest_addr = out_sp_addr,
                                  .size = (uint32_t)compute_size * _config.precision / _config.dram_req_size,
                                  .compute_size = (uint32_t)compute_size,
                                  .src_addrs = std::vector<addr_type>{
                                      act_sp_base_addr, weight_sp_addr},
                                  .tile_m = static_cast<unsigned int>(m_loop),
                                  .tile_k = static_cast<unsigned int>(c_loop),
                                  .tile_n = static_cast<unsigned int>(compute_size)
                                  }));
                }
              }
            }
          }
        }
      }
    }
  }
  /* MOVOUT at last iteration */
  if (tile->C == mapping.tile_out_loop.C - 1 &&
      tile->R == mapping.tile_out_loop.R - 1 &&
      tile->S == mapping.tile_out_loop.S - 1) {
    /* Pooling not fused */
    for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns++) {
      int N = tout_n_offset + Ns;
      for (int Qs = 0; Qs < mapping.tile_in_loop.Q  && Qs + tout_q_offset < mapping.total_loop.Q; Qs += q_loop_size) {
        int q_loop = std::min(mapping.total_loop.Q - tout_q_offset - Qs, static_cast<unsigned int>(q_loop_size));
        q_loop = std::min(static_cast<int>(mapping.tile_in_loop.Q) - Qs, q_loop);
        if(q_loop <= 0) break;
          for (int Ps = 0; Ps < mapping.tile_in_loop.P && Ps + tout_p_offset < mapping.total_loop.P; Ps += p_loop_size) {
            int p_loop = std::min(mapping.total_loop.P - tout_p_offset - Ps, static_cast<unsigned int>(p_loop_size));
            p_loop = std::min(static_cast<int>(mapping.tile_in_loop.P) - Ps, p_loop);
            if(p_loop <= 0) break;
            for (int Ms = 0; Ms < mapping.tile_in_loop.M  && Ms + tout_m_offset < mapping.total_loop.M; Ms += loop_size) {
              int m_loop = std::min(mapping.total_loop.M - tout_m_offset - Ms, static_cast<unsigned int>(loop_size));
              if(m_loop <= 0) break;
              addr_type out_sp_addr =
                  ACCUM_SPAD_BASE +
                  make_activation_address(
                      Ns, Qs, Ps, Ms,
                      std::vector<uint32_t>{
                          mapping.tile_in_loop.N, mapping.tile_in_loop.Q,
                          mapping.tile_in_loop.P, mapping.tile_in_loop.M});
              out_sp_addr = _config.align_address(out_sp_addr);

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
                          output_addr + make_activation_address(N, Q, P, M, _pool_out_shape));
                    else
                      out_dram_addrs.insert(
                          output_addr + make_activation_address(N, Q, P, M, _conv_out_shape));
                  }
                }
              }
              tile_idx++;
              if (_pool_fused) {
                tile->instructions.push_back(std::make_unique<Instruction>(
                    Instruction{.opcode = Opcode::MOVOUT_POOL,
                                .dest_addr = out_sp_addr,
                                .size = (uint32_t)out_dram_addrs.size(),
                                .src_addrs = std::vector<addr_type>(
                                    out_dram_addrs.begin(), out_dram_addrs.end()),
                                .operand_id = _OUTPUT_OPERAND}));
              } else {
                tile->instructions.push_back(std::make_unique<Instruction>(
                    Instruction{.opcode = Opcode::MOVOUT,
                                .dest_addr = out_sp_addr,
                                .size = (uint32_t)out_dram_addrs.size(),
                                .src_addrs = std::vector<addr_type>(
                                    out_dram_addrs.begin(), out_dram_addrs.end()),
                                .operand_id = _OUTPUT_OPERAND}));
              }
          }
        }
      }
    }
  }
  spdlog::trace("Layer {} Sram allocation size {} B act {} B weight {} B", _name,
                sram_allocation * _config.dram_req_size, act_allocation* _config.dram_req_size,
                (sram_allocation - act_allocation)* _config.dram_req_size);
  assert(sram_allocation * _config.dram_req_size <= _config.core_config[target_core].spad_size KB / 2);
  assert(act_allocation * _config.dram_req_size <= _config.core_config[target_core].spad_size KB / 2);
}

void ConvWS::initialize_matmul_instructions(Tile* tile) {
  std::vector<uint32_t> output_shape = _conv_out_shape;
  uint32_t kernel_size = _weight_shape[Sdim] * _weight_shape[Rdim];
  uint32_t channels = _input_shape[Cdim];
  uint32_t N_dim_size =
      output_shape[Ndim] * output_shape[Hdim] * output_shape[Wdim];
  uint32_t C_dim_size = kernel_size * _weight_shape[Cdim_w] / _group;
  int loop_size = _config.core_config[target_core].core_width;
  addr_type weight_offset =
      tile->M * kernel_size * (channels / _group) * _config.precision;

  addr_type act_sp_base = SPAD_BASE;
  addr_type weight_sp_base =
      SPAD_BASE + C_dim_size * N_dim_size * _config.precision;
  addr_type out_sp_base = ACCUM_SPAD_BASE;

  /*Bias */
  for (int M = tile->M; M < tile->M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile->M + (_weight_shape[Mdim] / _group)
                     ? tile->M + (_weight_shape[Mdim] / _group) - M
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
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = bias_sp_addr,
          .size = (uint32_t)skip_addrs.size() * n_loop,
          .src_addrs =
              std::vector<addr_type>(skip_addrs.begin(), skip_addrs.end()),
          .operand_id = _INPUT_OPERAND + 2}));
    }
  }

  /*Skip */
  for (int M = tile->M; M < tile->M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile->M + (_weight_shape[Mdim] / _group)
                     ? tile->M + (_weight_shape[Mdim] / _group) - M
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
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = skip_sp_addr,
          .size = (uint32_t)skip_addrs.size(),
          .src_addrs =
              std::vector<addr_type>(skip_addrs.begin(), skip_addrs.end()),
          .operand_id = _INPUT_OPERAND + 3}));
    }
  }

  for (int M = tile->M; M < tile->M + (_weight_shape[Mdim] / _group);
       M += loop_size) {
    int m_loop = M + loop_size > tile->M + (_weight_shape[Mdim] / _group)
                     ? tile->M + (_weight_shape[Mdim] / _group) - M
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
        if (M == tile->M) {
          std::set<addr_type> act_addr;
          for (int c_iter = 0; c_iter < c_loop; c_iter++) {
            for (int n_iter = 0; n_iter < n_loop; n_iter++) {
              act_addr.insert(make_activation_address(
                  N + n_iter, 0, 0, C + c_iter,
                  std::vector<uint32_t>{N_dim_size, 0, 0,
                                        C_dim_size * _group}));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .size = (uint32_t)act_addr.size(),
              .src_addrs =
                  std::vector<addr_type>(act_addr.begin(), act_addr.end()),
              .operand_id = _INPUT_OPERAND}));
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
          tile->instructions.push_back(std::make_unique<Instruction>(
              Instruction{.opcode = Opcode::MOVIN,
                          .dest_addr = weight_sp_addr,
                          .size = (uint32_t)weight_addr.size(),
                          .src_addrs = std::vector<addr_type>(
                              weight_addr.begin(), weight_addr.end()),
                          .operand_id = _INPUT_OPERAND + 1}));
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
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVOUT,
              .dest_addr = out_sp_addr,
              .size = (uint32_t)out_addrs.size(),
              .src_addrs =
                  std::vector<addr_type>(out_addrs.begin(), out_addrs.end()),
              .operand_id = _OUTPUT_OPERAND}));
        }
      }
    }
  }
}

addr_type ConvWS::make_activation_address(uint32_t N, uint32_t H, uint32_t W,
                                             uint32_t C,
                                             std::vector<uint32_t> shape) {
  addr_type address;
  if (_config.layout == "NCHW") {
    address = (N * shape[Cdim] * shape[Hdim] * shape[Wdim] +
               C * shape[Hdim] * shape[Wdim] + H * shape[Wdim] + W) *
              _config.precision;
  } else if (_config.layout == "NHWC") {
    address = (N * shape[Hdim] * shape[Wdim] * shape[Cdim] +
               H * shape[Wdim] * shape[Cdim] + W * shape[Cdim] + C) *
              _config.precision;
  }
  return _config.align_address(address);
}

addr_type ConvWS::make_weight_address(uint32_t S, uint32_t R, uint32_t M,
                                         uint32_t C,
                                         std::vector<uint32_t> shape) {
  addr_type address;
  int padded_C;
  if (shape[Cdim_w] % _config.core_config[target_core].core_width)
    padded_C = shape[Cdim_w] + (_config.core_config[target_core].core_width - shape[Cdim_w] % _config.core_config[target_core].core_width);
  else
    padded_C = shape[Cdim_w];

  if (_config.layout == "NCHW") {
    address = (M * shape[Cdim_w] * shape[Sdim] * shape[Rdim] +
               C * shape[Sdim] * shape[Rdim] + S * shape[Rdim] + R) *
              _config.precision;
  } else if (_config.layout == "NHWC") {
    address = ((M / _config.core_config[target_core].core_width) * shape[Sdim] * shape[Rdim] * padded_C +
               S * shape[Rdim] * padded_C + R * padded_C + C) *
              _config.precision;

  }
  return _config.align_address(address);
}
