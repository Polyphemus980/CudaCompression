#include "fixed-length_kernels.cuh"

static constexpr int BITS_IN_BYTE = 8;
static constexpr int FRAME_LENGTH = 128;
static constexpr int THREADS_PER_BLOCK = 1024;
static constexpr int FRAMES_PER_BLOCK = THREADS_PER_BLOCK / FRAME_LENGTH;

struct multiply_and_add {
	const uint32_t multiplier; 

	multiply_and_add(int _multiplier) : multiplier(_multiplier) {}

	__host__ __device__
		uint32_t operator()(uint32_t prev_sum, uint32_t current_value) const {
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
__global__ void calculateFrameBits(unsigned char* data,int length,uint32_t* frameBits){
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (threadId >= length)
		return; 

	__shared__ int maxes[FRAMES_PER_BLOCK];
	if (threadIdx.x < FRAMES_PER_BLOCK)
		maxes[threadIdx.x] = 0;
	
	__syncthreads();

	int bits = BITS_IN_BYTE - countLeadingZeros(data[threadId]);
	atomicMax(&maxes[threadIdx.x / FRAME_LENGTH], bits);

	__syncthreads();

	int elements_in_block = min(length - blockIdx.x * blockDim.x, blockDim.x);
	int frames_used = (elements_in_block + FRAME_LENGTH - 1) / FRAME_LENGTH;

	if (threadIdx.x < frames_used) {
		frameBits[blockIdx.x * FRAMES_PER_BLOCK + threadIdx.x] = maxes[threadIdx.x];
	}
}


__global__ void fillOutput(unsigned char* data, int length, uint32_t* frameBits, uint32_t* output,int* framePositions) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= length)
		return;

	int frameIndex = threadId / FRAME_LENGTH;

	uint32_t bitsPerSymbol = frameBits[frameIndex];

	int symbolIndex = threadIdx.x % FRAME_LENGTH;
	int startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;

	int outputIndex = startPos / 32;
	
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

int calculateOutputSize(const int* framePositions, int numFrames, std::vector<uint32_t> frameBits,uint64_t dataLength) {
	int lastFrameStart = framePositions[numFrames - 1];
	int lastFrameBitLength = frameBits[numFrames - 1];
	int lastFrameDataCount = dataLength % FRAME_LENGTH == 0 ? FRAME_LENGTH : dataLength % FRAME_LENGTH;
	int lastBitPosition = lastFrameStart + lastFrameBitLength * lastFrameDataCount;
	return (lastBitPosition + 31) / 32;
}

std::vector<unsigned char> convertToUnsignedChar(std::vector<uint32_t>& input) {
	std::vector<unsigned char> output;
	output.reserve(input.size());

	for (uint32_t value : input) {
		assert(value < 256 && "Value exceeds unsigned char range!");
		output.push_back(static_cast<unsigned char>(value));
	}

	return output;
}

__host__ FLData CudaFLEncode(std::vector<unsigned char> data) {
	unsigned char* dev_data = NULL;
	uint64_t length = data.size();

	cudaMalloc((void**)&dev_data, sizeof(char) * length);
	cudaMemcpy(dev_data, data.data(), sizeof(char) * length, cudaMemcpyHostToDevice);

	int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int numFrames = (length + FRAME_LENGTH - 1) / FRAME_LENGTH;

	uint32_t* dev_frameBits = NULL;
	cudaMalloc((void**)&dev_frameBits, sizeof(uint32_t) * numFrames);

	calculateFrameBits << <numBlocks, THREADS_PER_BLOCK >> > (dev_data, length, dev_frameBits);

	std::vector<uint32_t> host_frameBits(numFrames);
	cudaMemcpy(host_frameBits.data(), dev_frameBits, sizeof(uint32_t) * numFrames, cudaMemcpyDeviceToHost);

	int* dev_framePositions = NULL;
	cudaMalloc((void**)&dev_framePositions, sizeof(int) * numFrames);


	thrust::exclusive_scan(thrust::device, dev_frameBits, dev_frameBits + numFrames, dev_framePositions, 0, multiply_and_add(FRAME_LENGTH));

	int* framePositions = (int*)malloc(sizeof(int) * numFrames);
	cudaMemcpy(framePositions, dev_framePositions, sizeof(int) * numFrames, cudaMemcpyDeviceToHost);

	uint32_t* dev_output = NULL;
	int outputLength = calculateOutputSize(framePositions, numFrames, host_frameBits, length);
	cudaMalloc((void**)&dev_output, sizeof(uint32_t) * outputLength);
	cudaMemset(dev_output, 0, sizeof(uint32_t) * outputLength);

	fillOutput << <numBlocks, THREADS_PER_BLOCK >> > (dev_data, length, dev_frameBits, dev_output, dev_framePositions);

	std::vector<uint32_t> output(outputLength);
	cudaMemcpy(output.data(), dev_output, sizeof(uint32_t) * outputLength, cudaMemcpyDeviceToHost);

	FLData encodedData;

	encodedData.encodedValues = output;
	encodedData.frameBits = convertToUnsignedChar(host_frameBits);
	encodedData.valuesLength = encodedData.encodedValues.size();
	encodedData.bitsLength = encodedData.frameBits.size();
	encodedData.decodedDataLength = data.size();

	return encodedData;
}

__global__ void Decode(uint32_t* encodedData, int encodedLength, unsigned char* frameBits, uint64_t decodedDataLength,unsigned char* decoded, uint32_t* framePositions)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= decodedDataLength)
		return;
	
	int frameIndex = threadId / FRAME_LENGTH;

	int bitsPerSymbol = frameBits[frameIndex];

	int symbolIndex = threadIdx.x % FRAME_LENGTH;
	int startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;
	int outputIndex = startPos / 32;

	int bitOffset = startPos % 32;

	unsigned char data = (encodedData[outputIndex] >> bitOffset);

	if (bitOffset + bitsPerSymbol > 32)
	{
		int spilledBits = bitOffset + bitsPerSymbol - 32;
		unsigned char spilledData = (encodedData[outputIndex + 1] & ((1u << spilledBits) - 1)) << (32 - bitOffset);
		data |= spilledData;
	}

	data &= ((1u << bitsPerSymbol) - 1);
	decoded[threadId] = data;

}

__host__ std::vector<unsigned char> CudaFLDecode(FLData decodingData) {

	uint64_t encodedLength = decodingData.valuesLength;
	uint64_t frameBitsLength = decodingData.bitsLength;
	uint64_t decodedDataLength = decodingData.decodedDataLength;

	uint32_t* dev_encodedData = NULL;
	cudaMalloc((void**)&dev_encodedData, sizeof(uint32_t) * encodedLength);
	cudaMemcpy(dev_encodedData,decodingData.encodedValues.data(), sizeof(uint32_t) * encodedLength, cudaMemcpyHostToDevice);
	
	unsigned char* dev_frameBits = NULL;
	cudaMalloc((void**)&dev_frameBits, sizeof(unsigned char) * frameBitsLength);
	cudaMemcpy(dev_frameBits,decodingData.frameBits.data(), sizeof(unsigned char) * frameBitsLength, cudaMemcpyHostToDevice);

	uint32_t* dev_framePositions = NULL;
	cudaMalloc((void**)&dev_framePositions, sizeof(int) * frameBitsLength);

	uint32_t* framePositions = (uint32_t*)malloc(sizeof(uint32_t) * frameBitsLength);
	int sum = 0;
	for (int i = 0; i < frameBitsLength; i++) {
		framePositions[i] = sum;
		sum += decodingData.frameBits[i] * FRAME_LENGTH;
	}

	cudaMemcpy(dev_framePositions, framePositions, sizeof(uint32_t) * frameBitsLength, cudaMemcpyHostToDevice);
	int numBlocks = (decodedDataLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	unsigned char* dev_decoded;
	cudaMalloc((void**)&dev_decoded, sizeof(char) * decodedDataLength);
	cudaMemset(dev_decoded, 0, sizeof(char) * decodedDataLength);
	cudaDeviceSynchronize();
	Decode<<<numBlocks, THREADS_PER_BLOCK >>>(dev_encodedData, encodedLength, dev_frameBits, decodedDataLength, dev_decoded, dev_framePositions);
	cudaDeviceSynchronize();
	
	std::vector<unsigned char> host_decoded(decodedDataLength);
	cudaMemcpy(host_decoded.data(), dev_decoded, sizeof(char) * decodedDataLength, cudaMemcpyDeviceToHost);
	
	return host_decoded;
}