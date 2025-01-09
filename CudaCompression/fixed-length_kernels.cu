#include "fixed-length_kernels.cuh"

static constexpr int bits_in_byte = 8;
static constexpr int frame_length = 512;
static constexpr int threads_per_block = 1024;
static constexpr int frames_per_block = threads_per_block / frame_length;

struct multiply_and_add {
	const int multiplier; 

	multiply_and_add(int _multiplier) : multiplier(_multiplier) {}

	__host__ __device__
		int operator()(int prev_sum, int current_value) const {
		return prev_sum + multiplier * current_value;
	}
};
__device__ int countLeadingZeros(unsigned char data) {
	if (data == 0)
		return 7;
	int count = 0;
	while ((data & (1 << 7)) == 0)
	{
		count++;
		data <<= 1;
	}
	return count;
}
__global__ void calculateFrameBits(unsigned char* data,int length,int* frameBits){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (threadId >= length)
		return; 

	__shared__ int maxes[frames_per_block];
	if (threadIdx.x < frames_per_block)
		maxes[threadIdx.x] = 0;
	
	__syncthreads();

	int bits = bits_in_byte - countLeadingZeros(data[threadId]);
	atomicMax(&maxes[threadIdx.x / frame_length], bits);

	__syncthreads();

	int elements_in_block = min(length - blockIdx.x * blockDim.x, blockDim.x);
	int frames_used = (elements_in_block + frame_length - 1) / frame_length;

	if (threadIdx.x < frames_used) {
		frameBits[blockIdx.x * frames_per_block + threadIdx.x] = maxes[threadIdx.x];
	}
}


__global__ void fillOutput(unsigned char* data, int length, int* frameBits, int numFrames, uint32_t* output,int* framePositions) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= length)
		return;

	int frameInBlock = threadIdx.x / frame_length;
	int frameIndex = blockIdx.x * frames_per_block + frameInBlock;

	int bitsPerSymbol = frameBits[frameIndex];

	int symbolIndex = threadIdx.x % frame_length;
	// starting bit in outputData
	int startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;

	// dividing the start position by the number of bits in a uint32_t to get the index in output array
	int outputIndex = startPos / 32;
	
	// % bits in a uint32_t to get the offset from the first bit of data 
	int bitOffset = startPos % 32;

	unsigned char symbol = data[threadId];

	uint32_t maskedSymbol = (uint32_t)(symbol & ((1u << bitsPerSymbol) - 1));

	uint32_t shiftedSymbol = maskedSymbol << bitOffset;

	atomicOr(&output[outputIndex], shiftedSymbol);

	if (bitOffset + bitsPerSymbol > 32) {
		uint32_t spillBits = bitOffset + bitsPerSymbol - 32;
		uint32_t spillMask = maskedSymbol >> (bitsPerSymbol - spillBits);
		atomicOr(&output[outputIndex + 1], spillMask);
	}

}

int calculateOutputSize(const int* framePositions, int numFrames, int* frameBits,uint64_t dataLength) {
	int lastFrameStart = framePositions[numFrames - 1];
	int lastFrameBitLength = frameBits[numFrames - 1];
	if (lastFrameBitLength == 0)
	{
		return (lastFrameStart + 31) / 32;
	}
	int lastFrameDataCount = dataLength % frame_length == 0 ? frame_length : dataLength % frame_length;
	int lastBitPosition = lastFrameStart + lastFrameBitLength * lastFrameDataCount;
	return (lastBitPosition + 31) / 32;
}
__host__ void CudaFLEncoding(unsigned char* data, int length) {
	unsigned char* dev_data = NULL;
	cudaMalloc((void**)&dev_data, sizeof(char) * length);
	cudaMemcpy(dev_data, data, sizeof(char) * length, cudaMemcpyHostToDevice);

	int numBlocks = (length + threads_per_block - 1) / threads_per_block;
	int numFrames = (length + frame_length - 1)/frame_length;

	int* dev_frameBits = NULL;
	cudaMalloc((void**)&dev_frameBits, sizeof(int) * numFrames);

	calculateFrameBits<<<numBlocks,threads_per_block>>>(dev_data, length, dev_frameBits);

	int* host_frameBits = (int*)malloc(sizeof(int) * numFrames);
	cudaMemcpy(host_frameBits, dev_frameBits, sizeof(int) * numFrames, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < numFrames; i++) {
		std::cout << host_frameBits[i] << "\t";
	}
	/*int* dev_framePositions = NULL;
	cudaMalloc((void**)&dev_framePositions, sizeof(int) * numFrames);
	
	if (host_frameBits[numFrames - 1] == 0)
	{
		thrust::exclusive_scan(thrust::device, dev_frameBits, dev_frameBits + numFrames - 2, dev_framePositions, 0, multiply_and_add(frame_length));
		dev_framePositions[numFrames - 2] = 10;
	}
	thrust::exclusive_scan(thrust::device, dev_frameBits, dev_frameBits + numFrames, dev_framePositions, 0,multiply_and_add(frame_length));

	int* framePositions = (int*)malloc(sizeof(int) * numFrames);
	cudaMemcpy(framePositions, dev_framePositions, sizeof(int) * numFrames, cudaMemcpyDeviceToHost);

	uint32_t* dev_output = NULL;
	cudaMalloc((void**)&dev_output, sizeof(uint32_t) * calculateOutputSize(framePositions, numFrames, host_frameBits, length));*/
}