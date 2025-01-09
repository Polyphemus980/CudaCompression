#pragma once

#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cstdint>

#define MAX_BYTE_VALUE 255

__global__ void createMask(unsigned char* inputBytes, uint32_t* mask, int n);

__host__ void scanMask(uint32_t* base_mask, uint32_t* scanned_mask, int n);

__global__ void compactMask(uint32_t* scanned_mask, uint32_t* compacted_mask, int* compactedLength, int n);

__global__ void calculateChunks(uint32_t* compacted_mask, uint32_t* chunks, int* compactedLength);

__global__ void finalEncoding(uint32_t* compacted_mask, uint32_t* finalPositions, int* compactedLength, unsigned char* inputBytes, unsigned char* outBytes, unsigned char* outCounts);

void calculatePosition(uint32_t* counts, uint32_t* positions, int counts_length);

__global__ void finalDecoding(unsigned char* encodedValues, uint32_t* positions, int encodedLength, unsigned char* decoded, int decodedLength);

__host__ void CudaRLEncoding(unsigned char* data, int length);

__host__ void CudaRLDecoding(unsigned char* encodedValues, unsigned char* encodedCounts, int encodedLength, unsigned char** decoded, int* decodedLength);