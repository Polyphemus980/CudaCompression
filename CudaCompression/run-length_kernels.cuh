#pragma once

#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstdint>
#include <vector>

#define MAX_BYTE_VALUE 255


struct RLData {
	std::vector<unsigned char> values;
	std::vector<unsigned char> counts;
	uint64_t length;
};
__host__ RLData CudaRLEncode(std::vector<unsigned char> data);

__host__ std::vector<unsigned char> CudaRLDecode(RLData decodingData);
