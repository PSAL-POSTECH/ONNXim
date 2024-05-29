#ifndef ITER_LEVEL_SCHEDULER_H
#define ITER_LEVEL_SCHEDULER_H
#include "LanguageScheduler.h"

class IterLevelScheduler : public LangScheduler {
  public:
    IterLevelScheduler(std::string name, std::string path, 
                  std::unique_ptr<LanguageModel> model,
                  SimulationConfig config,
                  json scheduler_config);
    virtual void cycle() override;
};


#endif