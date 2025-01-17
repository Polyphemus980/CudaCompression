#include "fixed-length_kernels.cuh"

static constexpr int BITS_IN_BYTE = 8;
static constexpr int FRAME_LENGTH = 128;
static constexpr int THREADS_PER_BLOCK = 1024;
static constexpr int FRAMES_PER_BLOCK = THREADS_PER_BLOCK / FRAME_LENGTH;

struct multiply_and_add {
	uint32_t multiplier;
	multiply_and_add(uint32_t multiplier) : multiplier(multiplier) {}

	__device__ uint32_t operator()(const uint32_t& a, const uint32_t& b) const {
		return a + (b * multiplier);
	}
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::string errorMessage = "GPUassert: " + std::string(cudaGetErrorString(code)) + " " + file + " " + std::to_string(line);
		throw std::runtime_error(errorMessage);
	}
}

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
__global__ void calculateFrameBits(unsigned char* data,uint32_t length,uint64_t* frameBits){
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


__global__ void fillOutput(unsigned char* data, uint64_t length, uint64_t* frameBits, uint32_t* output,uint64_t* framePositions) {
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= length)
		return;

	int frameIndex = threadId / FRAME_LENGTH;

	uint32_t bitsPerSymbol = frameBits[frameIndex];

	int symbolIndex = threadId % FRAME_LENGTH;
	uint64_t startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;

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

uint64_t calculateOutputSize(uint64_t* framePositions, int numFrames, std::vector<uint64_t> frameBits,uint64_t dataLength) {
	uint64_t lastFrameStart = framePositions[numFrames - 1];
	int lastFrameBitLength = frameBits[numFrames - 1];
	int lastFrameDataCount = dataLength % FRAME_LENGTH == 0 ? FRAME_LENGTH : dataLength % FRAME_LENGTH;
	uint64_t lastBitPosition = lastFrameStart + lastFrameBitLength * lastFrameDataCount;
	return (lastBitPosition + 31) / 32;
}

std::vector<unsigned char> convertToUnsignedChar(std::vector<uint64_t>& input) {
	std::vector<unsigned char> output;
	output.reserve(input.size());

	for (uint64_t value : input) {
		assert(value < 256 && "Value exceeds unsigned char range!");
		output.push_back(static_cast<unsigned char>(value));
	}

	return output;
}

__global__ void multiply(uint64_t* framePositions,uint32_t length, int multiplier) {
	uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= length)
		return;
	framePositions[threadId] *= multiplier;
}
__host__ FLData CudaFLEncode(std::vector<unsigned char> data) {
	unsigned char* dev_data = NULL;
	uint64_t* dev_frameBits = NULL;
	uint64_t* dev_framePositions = NULL;
	uint32_t* dev_output = NULL;
	uint64_t length = data.size();
	uint64_t* framePositions = NULL;

	try {
		gpuErrchk(cudaMalloc((void**)&dev_data, sizeof(char) * length));
		gpuErrchk(cudaMemcpy(dev_data, data.data(), sizeof(char) * length, cudaMemcpyHostToDevice));

		int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		int numFrames = (length + FRAME_LENGTH - 1) / FRAME_LENGTH;

		gpuErrchk(cudaMalloc((void**)&dev_frameBits, sizeof(uint64_t) * numFrames));
		calculateFrameBits << <numBlocks, THREADS_PER_BLOCK >> > (dev_data, length, dev_frameBits);
		gpuErrchk(cudaGetLastError());

		std::vector<uint64_t> host_frameBits(numFrames);
		gpuErrchk(cudaMemcpy(host_frameBits.data(), dev_frameBits, sizeof(uint64_t) * numFrames, cudaMemcpyDeviceToHost));

		gpuErrchk(cudaMalloc((void**)&dev_framePositions, sizeof(uint64_t) * numFrames));

		thrust::exclusive_scan(thrust::device, dev_frameBits, dev_frameBits + numFrames, dev_framePositions);
		
		int numMultiplyBlocks = (numFrames + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		multiply << <numMultiplyBlocks, THREADS_PER_BLOCK >> > (dev_framePositions, numFrames,FRAME_LENGTH);
		framePositions = (uint64_t*)malloc(sizeof(uint64_t) * numFrames);
		if (!framePositions) {
			throw std::runtime_error("Failed to allocate host memory for frame positions.");
		}

		gpuErrchk(cudaMemcpy(framePositions, dev_framePositions, sizeof(uint64_t) * numFrames, cudaMemcpyDeviceToHost));

		uint64_t outputLength = calculateOutputSize(framePositions, numFrames, host_frameBits, length);
		gpuErrchk(cudaMalloc((void**)&dev_output, sizeof(uint32_t) * outputLength));
		gpuErrchk(cudaMemset(dev_output, 0, sizeof(uint32_t) * outputLength));

		fillOutput << <numBlocks, THREADS_PER_BLOCK >> > (dev_data, length, dev_frameBits, dev_output, dev_framePositions);
		gpuErrchk(cudaGetLastError());

		std::vector<uint32_t> output(outputLength);
		gpuErrchk(cudaMemcpy(output.data(), dev_output, sizeof(uint32_t) * outputLength, cudaMemcpyDeviceToHost));

		FLData encodedData;
		encodedData.encodedValues = output;
		encodedData.frameBits = convertToUnsignedChar(host_frameBits);
		encodedData.valuesLength = encodedData.encodedValues.size();
		encodedData.bitsLength = encodedData.frameBits.size();
		encodedData.decodedDataLength = data.size();

		free(framePositions);
		gpuErrchk(cudaFree(dev_data));
		gpuErrchk(cudaFree(dev_frameBits));
		gpuErrchk(cudaFree(dev_framePositions));
		gpuErrchk(cudaFree(dev_output));

		return encodedData;
	}
	catch (const std::runtime_error& e) {
		std::cerr << "Error: " << e.what() << std::endl;

		if (dev_data) gpuErrchk(cudaFree(dev_data));
		if (dev_frameBits) gpuErrchk(cudaFree(dev_frameBits));
		if (dev_framePositions) gpuErrchk(cudaFree(dev_framePositions));
		if (dev_output) gpuErrchk(cudaFree(dev_output));
		if (framePositions) free(framePositions);

		return FLData();
	}
}

__global__ void Decode(uint32_t* encodedData, int encodedLength, unsigned char* frameBits, uint64_t decodedDataLength,unsigned char* decoded, uint64_t* framePositions)
{
	uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= decodedDataLength)
		return;
	
	int frameIndex = threadId / FRAME_LENGTH;

	int bitsPerSymbol = frameBits[frameIndex];

	int symbolIndex = threadId % FRAME_LENGTH;
	uint64_t startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;
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
	uint32_t* dev_encodedData = NULL;
	unsigned char* dev_frameBits = NULL;
	uint64_t* dev_framePositions = NULL;
	unsigned char* dev_decoded = NULL;
	uint64_t* framePositions = NULL;

	try {
		uint64_t encodedLength = decodingData.valuesLength;
		uint64_t frameBitsLength = decodingData.bitsLength;
		uint64_t decodedDataLength = decodingData.decodedDataLength;

		gpuErrchk(cudaMalloc((void**)&dev_encodedData, sizeof(uint32_t) * encodedLength));
		gpuErrchk(cudaMemcpy(dev_encodedData, decodingData.encodedValues.data(), sizeof(uint32_t) * encodedLength, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&dev_frameBits, sizeof(unsigned char) * frameBitsLength));
		gpuErrchk(cudaMemcpy(dev_frameBits, decodingData.frameBits.data(), sizeof(unsigned char) * frameBitsLength, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&dev_framePositions, sizeof(uint64_t) * frameBitsLength));

		thrust::exclusive_scan(thrust::device, dev_frameBits, dev_frameBits + frameBitsLength, dev_framePositions,(uint32_t)0);

		int numMultiplyBlocks = (frameBitsLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		multiply << <numMultiplyBlocks, THREADS_PER_BLOCK >> > (dev_framePositions, frameBitsLength, FRAME_LENGTH);

		gpuErrchk(cudaMalloc((void**)&dev_decoded, sizeof(unsigned char) * decodedDataLength));
		gpuErrchk(cudaMemset(dev_decoded, 0, sizeof(unsigned char) * decodedDataLength));

		int numBlocks = (decodedDataLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		Decode << <numBlocks, THREADS_PER_BLOCK >> > (dev_encodedData, encodedLength, dev_frameBits, decodedDataLength, dev_decoded, dev_framePositions);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		std::vector<unsigned char> host_decoded(decodedDataLength);
		gpuErrchk(cudaMemcpy(host_decoded.data(), dev_decoded, sizeof(unsigned char) * decodedDataLength, cudaMemcpyDeviceToHost));

		free(framePositions);
		gpuErrchk(cudaFree(dev_encodedData));
		gpuErrchk(cudaFree(dev_frameBits));
		gpuErrchk(cudaFree(dev_framePositions));
		gpuErrchk(cudaFree(dev_decoded));

		return host_decoded;
	}
	catch (const std::runtime_error& e) {
		std::cerr << "Error: " << e.what() << std::endl;

		if (dev_encodedData) gpuErrchk(cudaFree(dev_encodedData));
		if (dev_frameBits) gpuErrchk(cudaFree(dev_frameBits));
		if (dev_framePositions) gpuErrchk(cudaFree(dev_framePositions));
		if (dev_decoded) gpuErrchk(cudaFree(dev_decoded));
		if (framePositions) free(framePositions);

		return std::vector<unsigned char>();
	}
}