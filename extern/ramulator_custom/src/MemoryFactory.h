#ifndef __MEMORY_FACTORY_H
#define __MEMORY_FACTORY_H

#include <map>
#include <string>
#include <cassert>
#include <memory>

#include "Memory.h"
#include "DRAM.h"
#include "Controller.h"
#include "Config.h"

using namespace std;

namespace ram
{
template <typename T>
class MemoryFactory {
public:
    static void extend_channel_width(T* spec, int cacheline)
    {
        int channel_unit = spec->prefetch_size * spec->channel_width / 8;
        int gang_number = cacheline / channel_unit;
        
        assert(gang_number >= 1 && 
            "cacheline size must be greater or equal to minimum channel width");
        
        assert(cacheline == gang_number * channel_unit &&
            "cacheline size must be a multiple of minimum channel width");
        
        spec->channel_width *= gang_number;
    }

    static std::unique_ptr<Memory<T>> populate_memory(RamulatorConfig& configs, 
                                                      T *spec, 
                                                      int channels, int ranks) {
        int& default_ranks = spec->org_entry.count[int(T::Level::Rank)];
        int& default_channels = spec->org_entry.count[int(T::Level::Channel)];

        if (default_channels == 0) default_channels = channels;
        if (default_ranks == 0) default_ranks = ranks;

        vector<Controller<T> *> ctrls;
        for (int c = 0; c < channels; c++){
            DRAM<T>* channel = new DRAM<T>(spec, T::Level::Channel);
            channel->id = c;
            channel->regStats("");
            ctrls.push_back(new Controller<T>(configs, channel));
        }
        return std::make_unique<Memory<T>>(configs, ctrls);
    }

    static void validate(int channels, int ranks, RamulatorConfig& configs) {
        assert(channels > 0 && ranks > 0);
    }

    static std::unique_ptr<MemoryBase> create(RamulatorConfig& configs, 
                                              int cacheline) {
        int channels = stoi(configs["channels"], NULL, 0);
        int ranks = stoi(configs["ranks"], NULL, 0);
        
        validate(channels, ranks, configs);

        const string& org_name = configs["org"];
        const string& speed_name = configs["speed"];

        T *spec = new T(org_name, speed_name);

        // Set channel width statically in the header file
        //extend_channel_width(spec, cacheline);

        return populate_memory(configs, spec, channels, ranks);
    }
};

// template <>
// MemoryBase *MemoryFactory<WideIO2>::create(RamulatorConfig& configs, int cacheline);
// template <>
// MemoryBase *MemoryFactory<SALP>::create(RamulatorConfig& configs, int cacheline);

} /*namespace ram*/

#endif /*__MEMORY_FACTORY_H*/
