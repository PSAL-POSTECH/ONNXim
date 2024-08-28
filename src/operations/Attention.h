#pragma once
//#include "../tensor/NPUTensor.h"
#include "Operation.h"
#include "GemmWS.h"

class Attention : public Operation {
   public:
    Attention(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    Attention(SimulationConfig config, Model* model, std::string name, std::map<std::string, std::string>& attributes, uint32_t target_core=0);
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
    std::vector<uint32_t> _projection_output_shape;

    GemmWS* _projection_node;
    uint32_t _seq;
    uint32_t _q_len;
    uint32_t _dmodel;
    uint32_t _nh;
    uint32_t _nkvh;
    uint32_t _dk;

    uint32_t _key_projection_id;
    uint32_t _query_projection_id;
    uint32_t _value_projection_id;
    /* For kv cache */
    bool onnx = false;
    bool has_kv_cache = false;
    bool use_fused = true;
    bool need_scale = false;

    std::vector<uint32_t> _heads_per_tile;
    std::vector<uint32_t> _tiles_per_head;
    std::vector<uint32_t> _scale_tiles_per_head;

    void calculate_loops();
    void calculate_loops(Mapping& mapping);

    //void initialize_tiles();
    //void initialize_instructions(Tile &tile, int req_idx, int head_idx, int num_heads);
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_onnx_tiles(MappingTable& mapping_table);
    void initialize_non_fused_tiles(MappingTable& mapping_table);
    void initialize_instructions(Tile* tile, Mapping mapping, int head_idx, int num_heads);
    void initialize_instructions(Tile* tile, int head_idx, int num_heads);

    void initialize_scale_instructions(Tile* tile, Mapping mapping, int head_idx, int num_tiles, int query_idx, int num_queries);
   protected:
    uint32_t sram_size_needed();
};