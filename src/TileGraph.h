#include "Common.h"
#include "helper/HelperFunctions.h"
#include "operations/Operation.h"
#include "Tensor.h"
#include "Mapping.h"
class TileGraph {
  public:
    TileGraph(std::string onnx_path);

    void initialize_tile(std::string op_type);

  private:
    void tile_generate();

    std::deque<Instruction> _instructions;
    std::vector<uint32_t> _tile_size;
    std::vector<uint32_t> _tile_stride;
    std::vector<uint32_t> _stride_list;
    std::deque<Tile> _tiles;
    onnx::ModelProto _model_proto;
    uint32_t _root_node_id;
    uint32_t _precision;
    uint32_t _cycle;
    uint64_t _base_addr;
};