#include "gtest/gtest.h"
#include "Mapping.h"

TEST(OSMappingParsingTest, BasicAssertions) {
  /* Parse mapping for output stationary accelerator */
  Mapping mapping("T N1 C128 M128 Q28 P28 S3 R3 - O P4 - I S3 R3 C128 P7 M4 Q28Y M32X");
  /*Total loop count check*/
  EXPECT_EQ(mapping.total_loop.N, 1);
  EXPECT_EQ(mapping.total_loop.C, 128);
  EXPECT_EQ(mapping.total_loop.M, 128);
  EXPECT_EQ(mapping.total_loop.Q, 28);
  EXPECT_EQ(mapping.total_loop.P, 28);
  EXPECT_EQ(mapping.total_loop.S, 3);
  EXPECT_EQ(mapping.total_loop.R, 3);

  /*Spatial parsing check*/
  EXPECT_EQ(mapping.spatial_M, 32);
  EXPECT_EQ(mapping.spatial_Q, 28);
  EXPECT_EQ(mapping.spatial_P, 1);
  EXPECT_EQ(mapping.spatial_C, 1);
  EXPECT_EQ(mapping.spatial_R, 1);
  EXPECT_EQ(mapping.spatial_S, 1);
  
}

TEST(WSMappingParsingTest, BasicAssertions) {
  /* Parse mapping for weight stationary accelerator */
  Mapping mapping("T N1 C64 M256 Q56 P56 S1 R1 - O C8 - I M32 Q28 C8Y M8X P14 Q2 P4");
  /*Total loop count check*/
  EXPECT_EQ(mapping.total_loop.N, 1);
  EXPECT_EQ(mapping.total_loop.C, 64);
  EXPECT_EQ(mapping.total_loop.M, 256);
  EXPECT_EQ(mapping.total_loop.Q, 56);
  EXPECT_EQ(mapping.total_loop.P, 56);
  EXPECT_EQ(mapping.total_loop.S, 1);
  EXPECT_EQ(mapping.total_loop.R, 1);

  /*Spatial parsing check*/
  EXPECT_EQ(mapping.spatial_M, 8);
  EXPECT_EQ(mapping.spatial_Q, 1);
  EXPECT_EQ(mapping.spatial_P, 1);
  EXPECT_EQ(mapping.spatial_C, 8);
  EXPECT_EQ(mapping.spatial_R, 1);
  EXPECT_EQ(mapping.spatial_S, 1);
  
}