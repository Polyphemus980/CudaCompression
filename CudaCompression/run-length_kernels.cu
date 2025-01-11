#include "run-length_kernels.cuh";

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		std::string errorMessage = "GPUassert: " + std::string(cudaGetErrorString(code)) + " " + file + " " + std::to_string(line);
		throw std::runtime_error(errorMessage);  
	}
}

static constexpr int max_byte_value = 255;

__global__ void createMask(unsigned char* inputBytes, uint32_t* mask, int n) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= n)
		return;
	if (threadId == 0)
	{
		mask[threadId] = 1;
		return;
	}
	mask[threadId] = inputBytes[threadId] == inputBytes[threadId-1] ? 0 : 1;
}

__host__ void scanMask(uint32_t* base_mask, uint32_t* scanned_mask,int n)
{
	thrust::inclusive_scan(thrust::device,base_mask, base_mask + n, scanned_mask);
}

__global__ void compactMask(uint32_t* scanned_mask, uint32_t* compacted_mask,int* compactedLength, int n)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= n)
		return;
	if (threadId == n - 1)
	{
		compacted_mask[scanned_mask[threadId]] = n;
		*compactedLength = scanned_mask[threadId];
	}
	if (threadId == 0)
	{
		compacted_mask[threadId] = 0;
	}
	else if (scanned_mask[threadId] != scanned_mask[threadId - 1])
	{
		compacted_mask[scanned_mask[threadId] - 1] = threadId;
	}
}

__global__ void calculateChunks(uint32_t* compacted_mask, uint32_t* chunks, int* compactedLength) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int shared_length;
	if (threadId == 0)
		shared_length = *compactedLength;

	__syncthreads(); 

	if (threadId >= shared_length)
		return;
	unsigned int runLength = compacted_mask[threadId + 1] - compacted_mask[threadId];
	chunks[threadId] = (runLength + max_byte_value - 1) / max_byte_value;
}


__global__ void finalEncoding(uint32_t* compacted_mask, uint32_t* finalPositions,int* compactedLength,unsigned char* inputBytes, unsigned char* outBytes, unsigned char* outCounts) {
	
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int shared_length;
	if (threadId == 0)
		shared_length = *compactedLength;

	__syncthreads();

	if (threadId >= shared_length)
		return;

	int runLength = compacted_mask[threadId + 1] - compacted_mask[threadId];
	int outputPosition = finalPositions[threadId];
	unsigned char inputByte = inputBytes[compacted_mask[threadId]];
	
	int chunkCount = 0;
	while (runLength > 0)
	{
		unsigned char chunkSize = runLength >= 255 ? 255 : runLength;
		outBytes[outputPosition + chunkCount] = inputByte;
		outCounts[outputPosition + chunkCount] = chunkSize;
		runLength -= chunkSize;
		chunkCount++;
	}
}

void calculatePosition(uint32_t* counts, uint32_t* positions, int counts_length) {
	thrust::exclusive_scan(thrust::device, counts, counts + counts_length, positions);
}

void calculatePosition(unsigned char* counts, uint32_t* positions, int counts_length) {
	thrust::exclusive_scan(thrust::device, counts, counts + counts_length, positions);
}

__host__ RLData CudaRLEncode(std::vector<unsigned char> data) {
	cudaFree(0); 

	unsigned char* dev_data = NULL;
	uint32_t* mask = NULL;
	uint32_t* scanned_mask = NULL;
	uint32_t* compacted_mask = NULL;
	int* compacted_length = NULL;
	uint32_t* chunks = NULL;
	uint32_t* positions = NULL;
	unsigned char* outputBytes = NULL;
	unsigned char* outputCounts = NULL;
	int* host_compacted_length = NULL;

	auto start = std::chrono::high_resolution_clock::now();
	try {
		size_t length = data.size();

		gpuErrchk(cudaMalloc((void**)&dev_data, sizeof(char) * length));
		gpuErrchk(cudaMemcpy(dev_data, data.data(), sizeof(char) * length, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&mask, sizeof(uint32_t) * length));
		int threadsPerBlock = 512;
		int numBlocksMasks = (length + threadsPerBlock - 1) / threadsPerBlock;
		createMask << <numBlocksMasks, threadsPerBlock >> > (dev_data, mask, length);


		gpuErrchk(cudaMalloc((void**)&scanned_mask, sizeof(uint32_t) * length));
		scanMask(mask, scanned_mask, length);

		gpuErrchk(cudaMalloc((void**)&compacted_mask, sizeof(uint32_t) * length));
		gpuErrchk(cudaMalloc((void**)&compacted_length, sizeof(int)));

		compactMask << <numBlocksMasks, threadsPerBlock >> > (scanned_mask, compacted_mask, compacted_length, length);

		host_compacted_length = (int*)malloc(sizeof(int));
		gpuErrchk(cudaMemcpy(host_compacted_length, compacted_length, sizeof(int), cudaMemcpyDeviceToHost));

		int numBlocksChunks = (*host_compacted_length + threadsPerBlock - 1) / threadsPerBlock;
		gpuErrchk(cudaMalloc((void**)&chunks, sizeof(uint32_t) * *host_compacted_length));
		calculateChunks << <numBlocksChunks, threadsPerBlock >> > (compacted_mask, chunks, compacted_length);

		gpuErrchk(cudaMalloc((void**)&positions, sizeof(uint32_t) * *host_compacted_length));
		calculatePosition(chunks, positions, *host_compacted_length);

		uint32_t outputLength = thrust::reduce(thrust::device, chunks, chunks + *host_compacted_length, 0);

		gpuErrchk(cudaMalloc((void**)&outputBytes, sizeof(unsigned char) * outputLength));
		gpuErrchk(cudaMalloc((void**)&outputCounts, sizeof(unsigned char) * outputLength));

		finalEncoding << <numBlocksChunks, threadsPerBlock >> > (compacted_mask, positions, compacted_length, dev_data, outputBytes, outputCounts);

		std::vector<unsigned char> hostOutputBytes(outputLength);
		gpuErrchk(cudaMemcpy(hostOutputBytes.data(), outputBytes, sizeof(unsigned char) * outputLength, cudaMemcpyDeviceToHost));

		std::vector<unsigned char> hostOutputCounts(outputLength);
		gpuErrchk(cudaMemcpy(hostOutputCounts.data(), outputCounts, sizeof(unsigned char) * outputLength, cudaMemcpyDeviceToHost));

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "GPU Total time: " << duration.count() << " ms" << std::endl;

		RLData encodedData;
		encodedData.counts = hostOutputCounts;
		encodedData.values = hostOutputBytes;
		encodedData.tablesLength = outputLength;
		encodedData.decodedDataLength = data.size();
		
		free(host_compacted_length);
		gpuErrchk(cudaFree(dev_data));
		gpuErrchk(cudaFree(mask));
		gpuErrchk(cudaFree(scanned_mask));
		gpuErrchk(cudaFree(compacted_mask));
		gpuErrchk(cudaFree(compacted_length));
		gpuErrchk(cudaFree(chunks));
		gpuErrchk(cudaFree(positions));
		gpuErrchk(cudaFree(outputBytes));
		gpuErrchk(cudaFree(outputCounts));

		return encodedData;
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;

		if (dev_data) gpuErrchk(cudaFree(dev_data));
		if (mask) gpuErrchk(cudaFree(mask));
		if (scanned_mask) gpuErrchk(cudaFree(scanned_mask));
		if (compacted_mask) gpuErrchk(cudaFree(compacted_mask));
		if (compacted_length) gpuErrchk(cudaFree(compacted_length));
		if (chunks) gpuErrchk(cudaFree(chunks));
		if (positions) gpuErrchk(cudaFree(positions));
		if (outputBytes) gpuErrchk(cudaFree(outputBytes));
		if (outputCounts) gpuErrchk(cudaFree(outputCounts));
		if (host_compacted_length) free(host_compacted_length);
		return RLData();
	}
}

__global__ void finalDecoding(unsigned char* encodedValues, uint32_t* positions,int encodedLength, unsigned char* decoded,int decodedLength) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= decodedLength)
		return;

	int left = 0;
	int right = encodedLength - 1;

	while (left <= right) {
		int mid = (left + right) / 2;

		if (threadId < positions[mid]) {
			right = mid - 1;
		}
		else if (mid + 1 < encodedLength && threadId >= positions[mid + 1]) {
			left = mid + 1;
		}
		else {
			decoded[threadId] = encodedValues[mid];
			return;
		}
	}
}
__host__ std::vector<unsigned char> CudaRLDecode(RLData decodingData)
{	
	cudaFree(0);
	unsigned char* dev_encodedValues = NULL;
	unsigned char* dev_encodedCounts = NULL;
	uint32_t* positions = NULL;
	unsigned char* dev_decoded = NULL;

	auto start = std::chrono::high_resolution_clock::now();
	try
	{
		uint64_t encodedLength = decodingData.tablesLength;

		gpuErrchk(cudaMalloc((void**)&dev_encodedValues, sizeof(char) * encodedLength));
		gpuErrchk(cudaMemcpy(dev_encodedValues, decodingData.values.data(), sizeof(char) * encodedLength, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&dev_encodedCounts, sizeof(char) * encodedLength));
		gpuErrchk(cudaMemcpy(dev_encodedCounts, decodingData.counts.data(), sizeof(char) * encodedLength, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&positions, sizeof(unsigned char) * encodedLength));
		calculatePosition(dev_encodedCounts, positions, encodedLength);

		gpuErrchk(cudaMalloc((void**)&dev_decoded, sizeof(char) * decodingData.decodedDataLength));

		int threadsPerBlock = 512;
		int numBlocks = (decodingData.decodedDataLength + threadsPerBlock - 1) / threadsPerBlock;

		finalDecoding << <numBlocks, threadsPerBlock >> > (dev_encodedValues, positions, encodedLength, dev_decoded, decodingData.decodedDataLength);
		gpuErrchk(cudaGetLastError()); 

		std::vector<unsigned char> decoded(decodingData.decodedDataLength);
		gpuErrchk(cudaMemcpy(decoded.data(), dev_decoded, sizeof(char) * decodingData.decodedDataLength, cudaMemcpyDeviceToHost));

		gpuErrchk(cudaFree(dev_encodedValues));
		gpuErrchk(cudaFree(dev_encodedCounts));
		gpuErrchk(cudaFree(positions));
		gpuErrchk(cudaFree(dev_decoded));

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "GPU Total time: " << duration.count() << " ms" << std::endl;

		return decoded;
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;

		if (dev_encodedValues) gpuErrchk(cudaFree(dev_encodedValues));
		if (dev_encodedCounts) gpuErrchk(cudaFree(dev_encodedCounts));
		if (positions) gpuErrchk(cudaFree(positions));
		if (dev_decoded) gpuErrchk(cudaFree(dev_decoded));

		return {};
	}
}
