#pragma once
#include "../Common.h"

class AddressAllocator {
  virtual addr_type allocate(std::vector<int> shape, uint32_t data_size) = 0;
};