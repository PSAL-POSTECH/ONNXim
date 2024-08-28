#include "GemmWS.h"

#include "../Model.h"

GemmWS::GemmWS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Gemm(config, model, node_proto, target_core) {}

GemmWS::GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, bool has_bias, uint32_t target_core)
        : GemmWS(config, model, node_proto, target_core) {
        this->has_bias = has_bias;
}

GemmWS::GemmWS(SimulationConfig config, MappingTable& mapping_table,
               std::vector<uint32_t> input_shape,
               std::vector<uint32_t> weight_shape,
               std::vector<uint32_t> output_shape,
               uint32_t target_core)
    : Gemm(config, mapping_table, input_shape, weight_shape, output_shape, target_core) {}

GemmWS::GemmWS(SimulationConfig config, Model* model, std::string name,
               std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Gemm(config, model, name, attributes, target_core) {
  has_bias = std::stoi(get_attribute("has_bias"));
}

void GemmWS::initialize_tiles(MappingTable& mapping_table) {
  Mapping::LoopCounts key{.N = _output_shape[_input_shape.size()-2 + Ndim] * _batch_size,
                          .C = _weight_shape[Cdim_w],
                          .M = _weight_shape[Mdim],
                          .S = 1,
                          .R = 1,
                          .Q = 1,
                          .P = 1,
                          .target_core = target_core};

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
  float total_flops = key.M / ((float) 1e3) * key.N / ((float) 1e3) * key.C / ((float) 1e3) * 2;
  float bias_flops = key.M / ((float) 1e3) * key.N / ((float) 1e3) / ((float) 1e3);
  if (has_bias) {
    total_flops += bias_flops;
  }
  spdlog::info("[GemmWs] Keys K = {}, N = {}, M = {}", key.C, key.N, key.M);
  float total_memory = (key.M * key.C + key.N * key.C + key.N * key.M) * _config.precision / ((float) 1e9);
  float bias_memory = key.M *_config.precision / ((float) 1e9);
  if (has_bias) {
    total_memory += bias_memory;
  }
  spdlog::info("[GemmWS]: total {} GFLOPs, {} GB", total_flops, total_memory);
  float theoretical_compute_time = total_flops / _config.max_systolic_flops(target_core);
  float theoretical_mem_time = total_memory / _config.max_dram_bandwidth();
  float theoretical_time = std::max(theoretical_compute_time, theoretical_mem_time);
  spdlog::info("[GemmWS]: Theoretical time(ms): {} Compute time: {} Memory time: {}",
               theoretical_time * 1e3, theoretical_compute_time * 1e3, theoretical_mem_time * 1e3);
}

void GemmWS::initialize_instructions(Tile* tile, Mapping mapping) {
  int tout_m_offset = tile->M * mapping.tile_in_loop.M;
  int tout_c_offset = tile->C * mapping.tile_in_loop.C;
  int tout_n_offset = tile->batch * mapping.tile_in_loop.N;
  int elems_per_access = _config.dram_req_size / _config.precision;

  addr_type act_sp_base_addr = SPAD_BASE;
  addr_type weight_sp_base_addr = SPAD_BASE + mapping.tile_in_loop.N *
                                                  mapping.tile_in_loop.C *
                                                  _config.precision;


  addr_type first_addr, second_addr, third_addr, output_addr;
  first_addr = get_operand_addr(_INPUT_OPERAND);
  second_addr = get_operand_addr(_INPUT_OPERAND+1);
  third_addr = get_operand_addr(_INPUT_OPERAND+2);
  output_addr = get_operand_addr(_OUTPUT_OPERAND);

  int loop_size = _config.core_config[target_core].core_height;
  int cloop_size = mapping.tile_in_loop.C;
  for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if(m_loop <= 0) break;
    /* MOVIN BIAS */
    if(!tile->accum && has_bias) { 
      std::vector<addr_type> bias_addrs;
      for (int iter_m = 0; iter_m < m_loop; iter_m+=elems_per_access) {
            int M = M_offset + iter_m;
            if (M >= mapping.total_loop.M) continue;
            bias_addrs.push_back(third_addr + _config.align_address(M * _config.precision));
      }
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = ACCUM_SPAD_BASE + Ms * _config.precision,
              .size = (uint32_t)bias_addrs.size(),
              .src_addrs = std::vector<addr_type>(bias_addrs.begin(), bias_addrs.end()),
              .operand_id = _INPUT_OPERAND + 2}));
    }
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs+= cloop_size) {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                      ? mapping.total_loop.C - C_offset
                      : cloop_size;
      /* MOVIN Weights */
      addr_type weight_sp_addr =
            weight_sp_base_addr +
            (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      std::set<addr_type> weight_set;
      for (int iter_m = 0; iter_m < m_loop; iter_m+=1) {
        for (int iter_c = 0; iter_c < c_in_loop; iter_c+=elems_per_access) {
          int C = C_offset + iter_c;
          int M = M_offset + iter_m;
          std::vector<uint32_t> weight_shape_2d;
          std::vector<uint32_t> index;
          weight_shape_2d.resize(2);
          index.resize(2);
          weight_shape_2d[1] = _weight_shape[Cdim_w];
          weight_shape_2d[0] = _weight_shape[Mdim]; 
          index[1] = C;
          index[0] = M;
          weight_set.insert(
              second_addr + make_address(index, weight_shape_2d));
        }
      }
      tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
          .opcode = Opcode::MOVIN,
          .dest_addr = weight_sp_addr,
          .size = (uint32_t)weight_set.size(),
          .src_addrs = std::vector<addr_type>(weight_set.begin(), weight_set.end()),
          .operand_id = _INPUT_OPERAND + 1,
          .tile_m = mapping.tile_in_loop.M,
          .tile_k = mapping.tile_in_loop.C}));

      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                          ? mapping.total_loop.N - N_offset
                          : loop_size;
        if(n_loop <= 0) break;
        addr_type act_sp_addr =
            act_sp_base_addr +
            (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;

        /* MOVIN Activation */
        if (Ms == 0) {
          std::set<addr_type> input_set;
          for (int iter_n = 0; iter_n < n_loop; iter_n++) {
            for (int iter_c = 0; iter_c < c_in_loop; iter_c+=elems_per_access) {
              uint32_t N = N_offset + iter_n;
              uint32_t C = C_offset + iter_c;
              std::vector<uint32_t> index;
              if (_input_shape.size()==3)
                index = {N/_input_shape.at(1), N%_input_shape.at(1), C};
              
              else
                index = {N, C};
              input_set.insert(
                  first_addr + make_address(index, _input_shape));
            }
          }
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::MOVIN,
              .dest_addr = act_sp_addr,
              .size = (uint32_t)input_set.size(),
              .src_addrs = std::vector<addr_type>(input_set.begin(), input_set.end()),
              .operand_id = _INPUT_OPERAND,
              .tile_k = mapping.tile_in_loop.C,
              .tile_n = mapping.tile_in_loop.N}));
        }
      }
    }
  }
  /* Compute */
  for(int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
    int M_offset = tout_m_offset + Ms;
    int m_loop = M_offset + loop_size > mapping.total_loop.M
                     ? mapping.total_loop.M - M_offset
                     : loop_size;
    if(m_loop <= 0) break;
    for (int Cs = 0; Cs < mapping.tile_in_loop.C; Cs+= cloop_size) {
      int C_offset = tout_c_offset + Cs;
      int c_in_loop = C_offset + cloop_size > mapping.total_loop.C
                      ? mapping.total_loop.C - C_offset
                      : cloop_size;
      addr_type weight_sp_addr =
            weight_sp_base_addr +
            (Ms * mapping.tile_in_loop.C + Cs) * _config.precision;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                          ? mapping.total_loop.N - N_offset
                          : loop_size;
        if(n_loop <= 0) break;
        addr_type act_sp_addr =
            act_sp_base_addr +
            (Ns * mapping.tile_in_loop.C + Cs) * _config.precision;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        for(int c_iter = 0; c_iter < c_in_loop; c_iter+=_config.core_config[target_core].core_height) {
          int c_iter_size = c_in_loop - c_iter > _config.core_config[target_core].core_height ? _config.core_config[target_core].core_height : c_in_loop - c_iter;
          tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
              .opcode = Opcode::GEMM_PRELOAD,
              .dest_addr = out_sp_addr,
              .size = (uint32_t)n_loop,
              .compute_size = (uint32_t)n_loop,
              .src_addrs =
                  std::vector<addr_type>{act_sp_addr, weight_sp_addr},
              .tile_m = static_cast<unsigned int>(m_loop),
              .tile_k = static_cast<unsigned int>(c_iter_size),
              .tile_n = static_cast<unsigned int>(n_loop)}));
        }
      }
    }
  }

  /* MOVOUT */
  if (tout_c_offset + mapping.tile_in_loop.C >= mapping.total_loop.C){
    for (int Ms = 0; Ms < mapping.tile_in_loop.M; Ms += loop_size) {
      int M_offset = tout_m_offset + Ms;
      int m_loop = M_offset + loop_size > mapping.total_loop.M
                      ? mapping.total_loop.M - M_offset
                      : loop_size;
      if(m_loop <= 0) break;
      for (int Ns = 0; Ns < mapping.tile_in_loop.N; Ns += loop_size) {
        int N_offset = tout_n_offset + Ns;
        int n_loop = N_offset + loop_size > mapping.total_loop.N
                          ? mapping.total_loop.N - N_offset
                          : loop_size;
        if(n_loop <= 0) break;
        addr_type out_sp_addr =
            ACCUM_SPAD_BASE +
            (Ns * mapping.tile_in_loop.M + Ms) * _config.precision;
        std::set<addr_type> output_set;
        for (int iter_n = 0; iter_n < n_loop; iter_n++) {
          for (int iter_m = 0; iter_m < m_loop; iter_m+=elems_per_access) {
            uint32_t N = N_offset + iter_n;
            uint32_t M = M_offset + iter_m;
            std::vector<uint32_t> index;
            if (_output_shape.size()==3)
                index = {N/_output_shape.at(1), N%_output_shape.at(1), M};
              else
                index = {N, M};
            output_set.insert(output_addr + make_address(index, _output_shape));
          }
        }
          /*MOVOUT result at the last loop*/
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = out_sp_addr,
            .size = (uint32_t)output_set.size(),
            .src_addrs = std::vector<addr_type>(output_set.begin(), output_set.end()),
            .operand_id = _OUTPUT_OPERAND}));
      }
    }
  }
}