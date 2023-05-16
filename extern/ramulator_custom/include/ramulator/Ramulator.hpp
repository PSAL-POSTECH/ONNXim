#ifndef __RAMULATOR_H
#define __RAMULATOR_H
#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include <functional>
#include <robin_hood.h>
namespace ram {
class MemoryBase;
class Request;
class Ramulator {
public:
  Ramulator(const std::string ConfigFilePath, uint32_t num_core, bool is_pim = false);
  ~Ramulator();
  void tick();
  bool isAvailable(int CtrlID, uint64_t Addr, bool IsWrite) const;
  bool isAvailable(uint64_t Addr, bool IsWrite) const;
  void push(int CtrlID, uint64_t Addr, bool IsWrite, uint32_t core_id, void* original_req);
  void push(uint64_t Addr, bool IsWrite, uint32_t core_id, void* original_req);
  bool isEmpty(int CtrlID) const;
  const void* top(int CtrlID) const;
  void pop(int CtrlID);
  int getAtomicBytes() const;
  int getNumChannels() const;
  int getChannel(uint64_t Addr) const;
  void print_stats();
private:
  std::unique_ptr<MemoryBase> MemBase;
  class OutputPendingQueue;
  std::vector<OutputPendingQueue> OutputPendingQueues;
  using CallbackMap =
    std::unordered_map<bool, std::function<void(const ram::Request&)>>;
  CallbackMap Callbacks;
  robin_hood::unordered_flat_set<int> hot_vids;
  bool is_pim;
  static std::unique_ptr<MemoryBase> createMemory(std::string ConfigFilePath, uint32_t num_core);
};
class Ramulator::OutputPendingQueue {
public:
  OutputPendingQueue(int Size);
  bool isAvailable() const;
  bool isAvailable(uint32_t count) const;
  bool isEmpty() const;
  void reserve();
  void push(void* original_req);
  const void* top() const;
  void pop();
private:
  const int Size;
  int NumReserved;
  std::queue<void*> PendingQueue;
};
} // end namespace
#endif
