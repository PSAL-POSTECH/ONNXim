#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;
  ::testing::InitGoogleTest(&argc, argv);
  try {
    const bool create_default_logger = false;
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }
  return status;
}