#include "ConvOS.h"

#include <robin_hood.h>

#include "../Model.h"
#include "../Tensor.h"

ConvOS::ConvOS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Conv(config, model, node_proto, target_core) {}

ConvOS::ConvOS(const Conv& src) : Conv(src) {}

/* TODO: handle depthwise convolutoin (Important) */
/* TODO: handle grouped convolutoin (less important) */
void ConvOS::initialize_tiles(MappingTable& mapping_table) {
  int tile_h_size = _config.core_config[target_core].core_height;
  int tile_w_size = _config.core_config[target_core].core_width;
  int precision = _config.precision;
  spdlog::trace("initialize_tile {} ", _name);
  std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  std::vector<uint32_t> weight_shape = get_input(1)->get_dims();

  Mapping::LoopCounts key{.N = output_shape[Ndim],
                          .C = weight_shape[Cdim_w],
                          .M = weight_shape[Mdim],
                          .S = weight_shape[Sdim],
                          .R = weight_shape[Rdim],
                          .Q = output_shape[Hdim],
                          .P = output_shape[Wdim],
                          .target_core = target_core};
  Mapping mapping;
  try {
    mapping = mapping_table.at(key);
  } catch (const std::out_of_range& e) {
    spdlog::error("Key not found: N: {} C: {} M: {} P: {} Q: {} S: {} R: {}",
      key.N, key.C, key.M, key.P, key.Q, key.S, key.R);
    std::exit(EXIT_FAILURE);
  }
  // Tiling
  for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
    for (uint32_t tile_q = 0; tile_q < mapping.tile_out_loop.Q; tile_q++) {
      for (uint32_t tile_p = 0; tile_p < mapping.tile_out_loop.P; tile_p++) {
        for (uint32_t tile_m = 0; tile_m < mapping.tile_out_loop.M; tile_m++) {
          std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
                                .status = Tile::Status::INITIALIZED,
                                .optype = "Conv",
                                .layer_id = _id,
                                .batch = N,
                                .Q = tile_q,
                                .P = tile_p,
                                .M = tile_m});
          _tiles.push_back(std::move(tile));
          initialize_instructions(_tiles.back().get(), mapping);
        }
      }
    }
  }
  assert(_tiles.size() > 0);
}

void ConvOS::initialize_instructions(Tile* tile, Mapping mapping) {
  // std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  // std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  // std::vector<uint32_t> weight_shape = get_input(1)->get_dims();
  // uint32_t tile_size =
  //     mapping.tile_in_loop.C * mapping.tile_in_loop.R * mapping.tile_in_loop.S;
  // int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  // int tout_q_offset = tile->Q * mapping.tile_in_loop.Q;
  // int tout_p_offset = tile->P * mapping.tile_in_loop.P;

  // robin_hood::unordered_map<std::string, Instruction> inst_map;

  // for (int tout_C = 0; tout_C < mapping.tile_out_loop.C; tout_C++) {
  //   int c_offset = mapping.tile_in_loop.C * tout_C;
  //   /* Initialize Weight address */
  //   for (int Ms = 0; Ms < mapping.tile_in_loop.M / mapping.spatial_M; Ms++) {
  //     int m_offset = tout_m_offset + Ms * mapping.spatial_M;
  //     std::set<addr_type> weight_set;
  //     for (int S = 0; S < mapping.tile_in_loop.S; S++) {
  //       for (int R = 0; R < mapping.tile_in_loop.R; R++) {
  //         for (int tin_C = 0; tin_C < mapping.tile_in_loop.C; tin_C++) {
  //           int C = c_offset + tin_C;
  //           for (int spatial_m = 0; spatial_m < mapping.spatial_M;
  //                spatial_m++) {
  //             int M = m_offset + spatial_m;
  //             weight_set.insert(make_weight_address(S, R, M, C, weight_shape));
  //           }
  //         }
  //       }
  //     }
  //     std::string id =
  //         fmt::format("WEIGHT-{}-{}-{}", tile->layer_id, c_offset, m_offset);
  //     inst_map[id] = Instruction{.opcode = Opcode::MOVIN,
  //                                .id = id,
  //                                .addrs = std::vector<addr_type>(
  //                                    weight_set.begin(), weight_set.end())};
  //   }
  //   /* Initialize input activation address */
  //   for (int Qs = 0; Qs < mapping.tile_in_loop.Q / mapping.spatial_Q; Qs++) {
  //     int q_offset = tout_q_offset + Qs * mapping.spatial_Q;
  //     for (int Ps = 0; Ps < mapping.tile_in_loop.P / mapping.spatial_P; Ps++) {
  //       int p_offset = tout_p_offset + Ps * mapping.spatial_P;
  //       std::set<addr_type> input_set;
  //       for (int S = 0; S < mapping.tile_in_loop.S; S++) {
  //         for (int R = 0; R < mapping.tile_in_loop.R; R++) {
  //           for (int tin_C = 0; tin_C < mapping.tile_in_loop.C; tin_C++) {
  //             int C = c_offset + tin_C;
  //             for (int spatial_q = 0; spatial_q < mapping.spatial_Q;
  //                  spatial_q++) {
  //               for (int spatial_p = 0; spatial_p < mapping.spatial_P;
  //                    spatial_p++) {
  //                 int Q = q_offset + spatial_q;
  //                 int P = p_offset + spatial_p;
  //                 int IH = _strides[0] * Q - (S - weight_shape[Sdim] / 2);
  //                 int IW = _strides[1] * P - (R - weight_shape[Rdim] / 2);
  //                 if (IH < 0 || IW < 0 || IH >= input_shape[Hdim] ||
  //                     IW >= input_shape[Wdim]) {
  //                   continue;
  //                 }
  //                 input_set.insert(make_activation_address(tile->batch, IH, IW,
  //                                                          C, input_shape));
  //               }
  //             }
  //           }
  //         }
  //       }
  //       assert(input_set.size() > 0);
  //       std::string id = fmt::format("ACT-{}-{}-{}-{}", tile->layer_id, c_offset,
  //                                    q_offset, p_offset);
  //       inst_map[id] = Instruction{.opcode = Opcode::MOVIN,
  //                                  .id = id,
  //                                  .addrs = std::vector<addr_type>(
  //                                      input_set.begin(), input_set.end())};
  //     }
  //   }

  //   for (int Ms = 0; Ms < mapping.tile_in_loop.M / mapping.spatial_M; Ms++) {
  //     int m_offset = tout_m_offset + Ms * mapping.spatial_M;
  //     for (int Qs = 0; Qs < mapping.tile_in_loop.Q / mapping.spatial_Q; Qs++) {
  //       int q_offset = tout_q_offset + Qs * mapping.spatial_Q;
  //       for (int Ps = 0; Ps < mapping.tile_in_loop.P / mapping.spatial_P;
  //            Ps++) {
  //         int p_offset = tout_p_offset + Ps * mapping.spatial_P;
  //         std::set<addr_type> output_set;
  //         for (int spatial_m = 0; spatial_m < mapping.spatial_M; spatial_m++) {
  //           for (int spatial_q = 0; spatial_q < mapping.spatial_Q;
  //                spatial_q++) {
  //             for (int spatial_p = 0; spatial_p < mapping.spatial_P;
  //                  spatial_p++) {
  //               int M = m_offset + spatial_m;
  //               int Q = q_offset + spatial_q;
  //               int P = p_offset + spatial_p;
  //               output_set.insert(
  //                   make_activation_address(tile->batch, Q, P, M, output_shape));
  //             }
  //           }
  //         }
  //         std::string output_id = fmt::format(
  //             "OUTPUT-{}-{}-{}-{}-{}", tile->layer_id, tout_C, Ms, Qs, Ps);
  //         std::vector<std::string> dependent_id = std::vector<std::string>{
  //             fmt::format("WEIGHT-{}-{}-{}", tile->layer_id, c_offset, m_offset),
  //             fmt::format("ACT-{}-{}-{}-{}", tile->layer_id, c_offset, q_offset,
  //                         p_offset)};
  //         if (tout_C > 0) {
  //           std::string prev_output_id =
  //               fmt::format("OUTPUT-{}-{}-{}-{}-{}", tile->layer_id,
  //                           (tout_C - 1) * mapping.tile_in_loop.C, m_offset,
  //                           q_offset, p_offset);
  //           inst_map[prev_output_id] =
  //               Instruction{.opcode = Opcode::MOVIN,
  //                           .tile_size = tile_size,
  //                           .id = prev_output_id,
  //                           .addrs = std::vector<addr_type>(output_set.begin(),
  //                                                           output_set.end())};
  //           dependent_id.push_back(prev_output_id);
  //         }
  //         for (std::string inst : dependent_id) {
  //           tile->instructions.push_back(inst_map[inst]);
  //         }
  //         tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
  //             .opcode = Opcode::GEMM,
  //             .tile_size = tile_size,
  //             .id = fmt::format("GEMM-{}-{}-{}-{}", tile->layer_id, c_offset,
  //                               m_offset, q_offset, p_offset),
  //             .dependent_ids = dependent_id}));
  //         tile->instructions.push_back(
  //             Instruction{.opcode = Opcode::MOVOUT,
  //                         .id = output_id,
  //                         .addrs = std::vector<addr_type>(output_set.begin(),
  //                                                         output_set.end())});
  //       }
  //     }
  //   }
  // }
}
