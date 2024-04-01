#pragma once
//#include "../tensor/NPUTensor.h"
#include "Operation.h"

class Attention : public Operation {
   public:
    Attention(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
    //std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    /* q,k,v shape : (nh,{1,l},dk) / (nh,{l,l+1},dk) / (nh,{l,l+1},dk) */
    std::vector<uint32_t> _query_shape;
    std::vector<uint32_t> _key_shape;
    std::vector<uint32_t> _value_shape;

    std::vector<uint32_t> _weight_shape;
    std::vector<uint32_t> _bias_shape;
    std::vector<uint32_t> _mask_shape;
    std::vector<uint32_t> _kv_cache_shape;
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;
    std::vector<uint32_t> _liner_output_shape;

    uint32_t _seq;
    uint32_t _q_len;
    uint32_t _dmodel;
    uint32_t _nh;
    uint32_t _dk;

    uint32_t _linear_output_id;
    /* For kv cache */
    bool has_kv_cache = false;
    bool use_fused = true;

    std::vector<uint32_t> _heads_per_tile;

    void calculate_loops();
    //void initialize_tiles();
    //void initialize_instructions(Tile &tile, int req_idx, int head_idx, int num_heads);
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_non_fused_tiles(MappingTable& mapping_table);
    void initialize_instructions(Tile* tile, Mapping mapping, int head_idx, int num_heads);
   protected:
    uint32_t sram_size_needed();
    addr_type make_address(std::vector<uint32_t> index, std::vector<uint32_t> dims);
};