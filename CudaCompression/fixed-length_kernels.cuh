#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstdint>
#include <vector>

struct FLData {
	std::vector<uint32_t> encodedValues;
	std::vector<unsigned char> frameBits;
	uint64_t decodedDataLength;
	uint64_t valuesLength;
	uint64_t bitsLength;
};

__host__ uint32_t* CudaFLEncoding(unsigned char* data, uint64_t length);
__host__ void CudaFLDecoding(uint32_t* encodedData, int encodedLength, uint32_t* frameBits, int frameBitsLength, uint64_t decodedDataLength);