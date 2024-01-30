#include "Common.h"
#include "helper/HelperFunctions.h"
#include "operations/Operation.h"
#include "Tensor.h"
#include "Mapping.h"
class TileGraph {
  public:
    TileGraph(std::string onnx_path, SimulationConfig config);

    void initialize_tile(std::string op_type);
    void set_finish() { _finish = true; };
    bool check_finish() { return _finish; }
    uint32_t get_id() { return _root_node_id; }
    std::deque<Tile> get_tiles() { return _tiles; }
  private:
    void _tile_generate();
    void _base_addr_update();
    void _tile_index_generate();

    std::deque<Instruction> _instructions;
    std::vector<uint32_t> _tile_size;
    std::vector<uint32_t> _tile_stride;
    std::vector<uint32_t> _stride_list;
    std::vector<uint32_t> _tile_index;
    std::vector<uint64_t> _start;
    std::vector<uint64_t> _end;
    std::vector<uint64_t> _stride;
    std::deque<Tile> _tiles;
    SimulationConfig _config;
    onnx::ModelProto _model_proto;
    uint32_t _root_node_id;
    uint32_t _precision;
    uint32_t _cycle;
    addr_type _base_addr = 0;
    std::string _base_addr_ptr;
    addr_type _dest_addr;
    std::map<std::string, addr_type> _base_addr_map;
    std::vector<addr_type> _src_addrs;

    bool _finish;
};