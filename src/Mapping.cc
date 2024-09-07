#include "Mapping.h"

#include <fstream>
#include <sstream>
#include <string>
// #include "Common.h"

std::string Mapping::to_string() {
  std::stringstream ss;
  ss << "Total Loop: N" << total_loop.N << " C" << total_loop.C << " M" << total_loop.M
     << " S" << total_loop.S << " R" << total_loop.R << " Q" << total_loop.Q << " P" << total_loop.P
     << " Tile Out Loop: N" << tile_out_loop.N << " C" << tile_out_loop.C << " M" << tile_out_loop.M
     << " S" << tile_out_loop.S << " R" << tile_out_loop.R << " Q" << tile_out_loop.Q << " P" << tile_out_loop.P
     << " Tile In Loop: N" << tile_in_loop.N << " C" << tile_in_loop.C << " M" << tile_in_loop.M
     << " S" << tile_in_loop.S << " R" << tile_in_loop.R << " Q" << tile_in_loop.Q << " P" << tile_in_loop.P;
  return ss.str();
}

MappingTable::MappingTable () {}
MappingTable::MappingTable (SimulationConfig config) {
  _mapping_table = _MappingTable();
  _config = config;
}

MappingTable MappingTable::parse_mapping_file(
    std::string mapping_path, SimulationConfig config) {
  MappingTable map = MappingTable(config);
  std::ifstream mapping_file;
  std::string line;
  mapping_file.open(mapping_path);
  if (mapping_file.fail()) {
    spdlog::info("No mapping file path : {}", mapping_path);
    return map;
  }
  while (getline(mapping_file, line)) {
    Mapping mapping(line);
    map[mapping.total_loop] = mapping;
    spdlog::trace("N {} C {} M {} S {} R {} Q {} P {}",
      mapping.total_loop.N,
      mapping.total_loop.C,
      mapping.total_loop.M,
      mapping.total_loop.S,
      mapping.total_loop.R,
      mapping.total_loop.Q,
      mapping.total_loop.P
      );
  }
  mapping_file.close();
  return map;
}

void MappingTable::gemm_mapping(Mapping::LoopCounts &key) {
  uint32_t dim_I, dim_J, dim_K;
  uint32_t dim = _config.core_config[key.target_core].core_height;
  uint32_t max_spad_rows = (_config.core_config[key.target_core].spad_size KB) / (dim * _config.precision * 2);
  uint32_t max_acc_rows = (_config.core_config[key.target_core].accum_spad_size KB) / (dim * 4 * 2);

  assert(_config.core_config[key.target_core].core_height==_config.core_config[key.target_core].core_width);
  dim_I = key.N;
  dim_J = key.M;
  dim_K = key.C;

  const uint32_t dim_I_padded = (dim_I / dim + (dim_I % dim != 0 )) * dim;
  const uint32_t dim_J_padded = (dim_J / dim + (dim_J % dim != 0 )) * dim;
  const uint32_t dim_K_padded = (dim_K / dim + (dim_K % dim != 0 )) * dim;

  uint32_t tile_I, tile_J, tile_K;
  uint32_t inner_I, inner_J, inner_K;
  uint32_t db_partitions_rows, db_mats_in_partition, db_mats_in_acc;
  uint32_t db_max_tile_i_j, db_max_tile_k;

  db_partitions_rows = max_spad_rows / 2;
  db_mats_in_partition = db_partitions_rows / dim;
  db_mats_in_acc = max_acc_rows / dim;
  db_max_tile_i_j = (uint32_t)sqrt(db_mats_in_acc);
  db_max_tile_k = db_mats_in_partition / db_max_tile_i_j;

  tile_I = std::min(dim_I_padded/dim, ceil_div(dim_I, db_max_tile_i_j*dim));
  tile_J = std::min(dim_J_padded/dim, ceil_div(dim_J, db_max_tile_i_j*dim));
  tile_K = std::min(dim_K_padded/dim, ceil_div(dim_K, db_max_tile_k*dim));
  
  uint32_t num_tiles = tile_I * tile_J; //Skip C dim that needs accum
  if(num_tiles < _config.num_cores) {
    int increase_tile = ceil_div(_config.num_cores, num_tiles);
    if(dim_J > dim_I && dim_J > _config.num_cores) {
      tile_J *= increase_tile;
    } else if(dim_I > dim_J && dim_I > _config.num_cores) {
      tile_I *= increase_tile;
    }
    num_tiles = tile_I * tile_J;
  }
  if(num_tiles % _config.num_cores != 0) {
    int increase_tile = num_tiles % _config.num_cores;
    if(dim_J > dim_I && dim_J > _config.num_cores) {
      tile_J += increase_tile;
    } else if(dim_I > dim_J && dim_I > _config.num_cores) {
      tile_I += increase_tile;
    }
  }

  inner_I = ceil_div(dim_I_padded, tile_I);
  inner_J = ceil_div(dim_J_padded, tile_J);
  inner_K = ceil_div(dim_K_padded, tile_K);

  inner_I -= inner_I & (dim)-1;
  inner_J -= inner_J & (dim)-1;
  inner_K -= inner_K & (dim)-1;

  tile_I = ceil_div(dim_I, inner_I);
  tile_J = ceil_div(dim_J, inner_J);
  tile_K = ceil_div(dim_K, inner_K);

  /* create mapping entry */
  Mapping mapping;
  mapping.total_loop = {dim_I, dim_K, dim_J, 1, 1, 1, 1};
  mapping.tile_out_loop = {tile_I, tile_K, tile_J, 1, 1, 1, 1};
  mapping.tile_in_loop = {inner_I, inner_K, inner_J, 1, 1, 1, 1};
  _mapping_table[key] = mapping;
  spdlog::info("[GEMM] spad_size: {} accum_size: {}", _config.core_config[key.target_core].spad_size * 1024, _config.core_config[key.target_core].accum_spad_size * 1024);
  spdlog::info("[GEMM] required_sram_size: {} required_accum_size: {}", (inner_I+inner_J)*inner_K*_config.precision, (inner_I*inner_J)*_config.precision);
  spdlog::info("[GEMM] Used gemmini gemm mapping: Total N:{} C:{} M:{}, " \
    "Outer N:{} C:{} M:{}, " \
    "Inner N:{} C:{} M:{}",
    mapping.total_loop.N, mapping.total_loop.C, mapping.total_loop.M,
    mapping.tile_out_loop.N, mapping.tile_out_loop.C, mapping.tile_out_loop.M,
    mapping.tile_in_loop.N, mapping.tile_in_loop.C, mapping.tile_in_loop.M
  );
}

const Mapping& MappingTable::fallback_mapping(Mapping::LoopCounts &key) {
  if (key.P==1 && key.Q==1 && key.S==1 && key.R==1)
    gemm_mapping(key);
  else if (key.P==key.Q && key.S==key.R)
    conv_mapping(key);
  return _mapping_table.at(key);
}

const Mapping& MappingTable::at(Mapping::LoopCounts &key) {
  auto it = _mapping_table.find(key);
  if (it != _mapping_table.end())
    return it->second;
  else
    return fallback_mapping(key);
}

uint32_t Mapping::LoopCounts::get_loop(Mapping::LoopName name) {
  switch (name) {
    case Mapping::LoopName::N:
      return N;
    case Mapping::LoopName::C:
      return C;
    case Mapping::LoopName::M:
      return M;
    case Mapping::LoopName::S:
      return S;
    case Mapping::LoopName::R:
      return R;
    case Mapping::LoopName::P:
      return P;
    case Mapping::LoopName::Q:
      return Q;
    default:
      assert(0);
      /* Unreachable */
      return 0;
  }
}

Mapping::Mapping(std::string mapping_line) {
  std::string total_tile;
  std::string out_tile;
  std::string in_tile;
  std::string loop_elem;
  std::stringstream level_parse;
  std::stringstream loop_parse;
  spatial_M = 1;
  spatial_P = 1;
  spatial_Q = 1;
  spatial_C = 1;
  spatial_R = 1;
  spatial_S = 1;
  level_parse << mapping_line;

  // Parse Total Loop
  getline(level_parse, total_tile, '-');
  loop_parse << total_tile;
  getline(loop_parse, loop_elem, ' ');
  while (getline(loop_parse, loop_elem, ' ')) {
    LoopName loop_name;
    int loop_count = std::stoi(loop_elem.substr(1));
    switch (loop_elem.at(0)) {
      case 'N':
        total_loop.N = loop_count;
        break;
      case 'C':
        total_loop.C = loop_count;
        break;
      case 'M':
        total_loop.M = loop_count;
        break;
      case 'S':
        total_loop.S = loop_count;
        break;
      case 'R':
        total_loop.R = loop_count;
        break;
      case 'Q':
        total_loop.Q = loop_count;
        break;
      case 'P':
        total_loop.P = loop_count;
        break;
      default:
        assert(0);
    }
  }

  // Parse Outer Loop
  getline(level_parse, out_tile, '-');
  loop_parse.clear();
  loop_parse << out_tile;
  getline(loop_parse, loop_elem, ' ');
  getline(loop_parse, loop_elem, ' ');
  while (getline(loop_parse, loop_elem, ' ')) {
    LoopName loop_name;
    int loop_count = std::stoi(loop_elem.substr(1));
    switch (loop_elem.at(0)) {
      case 'N':
        loop_name = LoopName::N;
        tile_out_loop.N = loop_count;
        break;
      case 'C':
        loop_name = LoopName::C;
        tile_out_loop.C = loop_count;
        break;
      case 'M':
        loop_name = LoopName::M;
        tile_out_loop.M = loop_count;
        break;
      case 'S':
        loop_name = LoopName::S;
        tile_out_loop.S = loop_count;
        break;
      case 'R':
        loop_name = LoopName::R;
        tile_out_loop.R = loop_count;
        break;
      case 'Q':
        loop_name = LoopName::Q;
        tile_out_loop.Q = loop_count;
        break;
      case 'P':
        loop_name = LoopName::P;
        tile_out_loop.P = loop_count;
        break;
      default:
        assert(0);
    }
    tile_out_loop_order.push_back(loop_name);
  }

  // Parse L4

  while (getline(level_parse, in_tile, '-')) {
    loop_parse.clear();
    loop_parse << in_tile;
    getline(loop_parse, loop_elem, ' ');
    getline(loop_parse, loop_elem, ' ');
    while (getline(loop_parse, loop_elem, ' ')) {
      LoopName loop_name;
      int loop_count;
      switch (loop_elem.at(0)) {
        case 'N':
          tile_in_loop.N *= std::stoi(loop_elem.substr(1));
          break;
        case 'C':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_C = loop_count;
          }
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.C *= loop_count;
          break;
        case 'M':
          if (loop_elem.find('X') != std::string::npos) {
            size_t pos = loop_elem.find('X');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_M = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.M *= loop_count;
          break;
        case 'S':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_S = loop_count;
          }
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.S *= loop_count;
          break;
        case 'R':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_R = loop_count;
          }
          else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.R *= loop_count;
          break;
        case 'Q':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_Q = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.Q *= loop_count;
          break;
        case 'P':
          if (loop_elem.find('Y') != std::string::npos) {
            size_t pos = loop_elem.find('Y');
            loop_count = std::stoi(loop_elem.substr(1, pos));
            spatial_P = loop_count;
          } else {
            loop_count = std::stoi(loop_elem.substr(1));
          }
          tile_in_loop.P *= loop_count;
          break;
        default:
          assert(0);
      }
    }
  }
}

void MappingTable::conv_mapping(Mapping::LoopCounts &key) {
  auto mapping = calc_conv_mapping(key);
  _mapping_table[key] = mapping;
}

int MappingTable::_calc_conv_mapping(bool acc,
		int stride,
		int input_dilation,
		int kernel_dilation,
		bool downsample,
		bool trans_weight_0132,
		bool trans_input_3120,
		int batches,
		int porows, int pocols, int ochs,
		int krows, int kcols, int kchs,
		int pool_size, int pool_stride, int dim) {

	const int orows = porows * pool_stride + pool_size - 1;
	const int ocols = pocols * pool_stride + pool_size - 1;

	const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
	const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

	int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
	int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
	const int ichs = kchs;

	irows = irows / input_dilation + (irows % input_dilation != 0);
	icols = icols / input_dilation + (icols % input_dilation != 0);

	const int in_channels_per_bank = ichs / dim + (ichs % dim != 0);
	const int out_channels_per_bank = ochs / dim + (ochs % dim != 0);
	const int batches_per_bank = batches / dim + (batches % dim != 0);

	const int A_rows = trans_input_3120 ?
		(batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
		(in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

	const int B_rows = trans_weight_0132 ?
	  in_channels_per_bank * kcols * krows * ochs :
	  out_channels_per_bank * kcols * krows * kchs;

	const int C_rows = out_channels_per_bank * batches * orows * ocols;

	return acc ? C_rows : A_rows + B_rows;
}

Mapping MappingTable::calc_conv_mapping(Mapping::LoopCounts &key) {
  int batch_size, in_dim, in_channels;
  int out_channels, out_dim;
  int stride, input_dilation, kernel_dilation, padding, kernel_dim;
  bool trans_input_3120, trans_weight_0132;
  int pool_size, pool_stride, pool_padding;
  uint32_t dim = _config.core_config[key.target_core].core_height;
  uint32_t max_spad_rows = (_config.core_config[key.target_core].spad_size KB) / (dim * _config.precision * 2);
  uint32_t max_acc_rows = (_config.core_config[key.target_core].accum_spad_size KB) / (dim * 4 * 2);

  batch_size = 1;
  out_channels = key.M;
  in_channels = key.C;
  out_dim = key.P;
  kernel_dim = key.S;

  stride = key.Stride;
  input_dilation = 1;
  kernel_dilation = 1;
  padding = key.Padding;
  trans_input_3120 = false;
  trans_weight_0132 = false;
  pool_stride = 0;

	const bool no_pool = pool_stride == 0;
	if (no_pool) {
		pool_size = 1;
		pool_stride = 1;
		pool_padding = 0;
	}

	const int pool_out_dim = (out_dim + 2*pool_padding - pool_size) / pool_stride + 1;

	const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool;

	// Tile convolution params

	// int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
	int args[] = {batch_size, pool_out_dim, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};
	const int max_args[] = {batch_size, pool_out_dim, pool_out_dim, out_channels, kernel_dim, kernel_dim, in_channels};

	const int orows_idx = 1;
	const int ocols_idx = 2;
	const int out_channels_idx = 3;
	const int in_channels_idx = 6;

	int spad_rows = _calc_conv_mapping(false,
		stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
		args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride, dim);
	int acc_rows = _calc_conv_mapping(true,
		stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
		args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride, dim);

	while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
		int max_val = -1;
		int max_idx = -1;

		for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
			// We avoid reducing ocols when possible to keep the spatial array fully utilized
			if (!(i == ocols_idx && args[i] <= dim && args[orows_idx] > 1)
					&& args[i] > max_val) {
				max_val = args[i];
				max_idx = i;
			}
		}

		if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
			// For input and output channels, there's no point in subtracting by just one
			if (args[max_idx] % dim != 0) {
				args[max_idx] = (args[max_idx] / dim) * dim;
			} else {
				args[max_idx] -= dim;
			}
			args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
		} else {
			args[max_idx]--;
		}

		spad_rows = _calc_conv_mapping(false,
			stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
			args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride, dim);
		acc_rows = _calc_conv_mapping(true,
			stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
			args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride, dim);
	}

	// Check if we can increase ocols
	bool not_increased = false;
	while (!not_increased) {
		not_increased = true;

		int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
		args_candidate[ocols_idx]++;

		if (args_candidate[ocols_idx] > max_args[ocols_idx])
			continue;

		spad_rows = _calc_conv_mapping(false,
			stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
			args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride, dim);
		acc_rows = _calc_conv_mapping(true,
			stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
			args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride, dim);

		if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
			args[ocols_idx] = args_candidate[ocols_idx];
			not_increased = false;
		}
	}

	// Check if there are any parameters that we can currently still increase
	bool nothing_increased = false;
	while (!nothing_increased) {
		nothing_increased = true;

		for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
			int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
      args_candidate[i]++;
			if (args_candidate[i] > max_args[i])
				continue;

			spad_rows = _calc_conv_mapping(false,
				stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
				args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride, dim);
			acc_rows = _calc_conv_mapping(true,
				stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
				args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride, dim);
      // B * O_ch * K_row * K_col * K_ch
      int weight_tile_size = args[0] * args[3] * args[4] * args[5] * args[6] * _config.precision;
      // B * O_row * O_col * O_ch
      int input_tile_size = args[0] * (args[1]+2*padding) * (args[2]+2*padding) * args[6] * _config.precision;
			if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows &&
        ((input_tile_size + weight_tile_size) * 3 >> 1) <= (_config.core_config[key.target_core].spad_size KB / 2)) {
				args[i] = args_candidate[i];
				nothing_increased = false;
			}
		}
	}

	const int batches = args[0];
	const int orows = args[1];
	const int ocols = args[2];
	const int ochs = args[3];
	const int krows = args[4];
	const int kcols = args[5];
	const int kchs = args[6];
  batch_size = key.N;
	Mapping mapping;
	mapping.total_loop = {(uint32_t)batch_size, (uint32_t)in_channels, (uint32_t)out_channels,
                        (uint32_t)kernel_dim, (uint32_t)kernel_dim, (uint32_t)out_dim, (uint32_t)out_dim};
	mapping.tile_out_loop = {ceil_div(batch_size, batches), ceil_div(in_channels, kchs),
              ceil_div(out_channels, ochs), ceil_div(kernel_dim, krows),
							ceil_div(kernel_dim, kcols), ceil_div(out_dim, ocols),
							ceil_div(out_dim, orows)};
	mapping.tile_in_loop = {(uint32_t)batches, (uint32_t)kchs, (uint32_t)ochs,
                          (uint32_t)krows, (uint32_t)kcols, (uint32_t)ocols, (uint32_t)orows};
	spdlog::info("[Conv] Used gemmini convolution mapping: " \
		"[T] N{} C{} M{} P{} Q{} S{} R{} - " \
		"[O] N{} C{} M{} P{} Q{} S{} R{} - " \
		"[I] N{} C{} M{} P{} Q{} S{} R{}",
		batch_size, in_channels, out_channels, out_dim, out_dim, kernel_dim, kernel_dim,
		ceil_div(batch_size, batches), ceil_div(in_channels, kchs), ceil_div(out_channels, ochs),
		ceil_div(out_dim, orows), ceil_div(out_dim, ocols),
		ceil_div(kernel_dim, krows), ceil_div(kernel_dim, kcols),
		batches, kchs, ochs, orows, ocols, krows, kcols
	);
	return mapping;
}