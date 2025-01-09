#include "run-length_kernels.cuh";

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

__host__ void CudaRLEncoding(unsigned char* data, int length) {
	cudaFree(0);
	auto start = std::chrono::high_resolution_clock::now();
	unsigned char* dev_data = NULL;
	cudaMalloc((void**)&dev_data, sizeof(char) * length);
	cudaMemcpy(dev_data, data, sizeof(char) * length, cudaMemcpyHostToDevice);

	uint32_t* mask = NULL;
	cudaMalloc((void**)&mask, sizeof(uint32_t) * length);

	int threadsPerBlock = 512;
	int numBlocksMasks = (length + threadsPerBlock - 1) / threadsPerBlock;
	createMask << <numBlocksMasks, threadsPerBlock >> > (dev_data, mask, length);

	uint32_t* host_mask = (uint32_t*)malloc(sizeof(uint32_t) * length);
	cudaMemcpy(host_mask, mask, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost);

	uint32_t* scanned_mask = NULL;
	cudaMalloc((void**)&scanned_mask, sizeof(uint32_t) * length);

	scanMask(mask, scanned_mask, length);
	uint32_t* host_scanned_mask = (uint32_t*)malloc(sizeof(uint32_t) * length);
	cudaMemcpy(host_scanned_mask, scanned_mask, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost);

	uint32_t* compacted_mask = NULL;
	cudaMalloc((void**)&compacted_mask, sizeof(uint32_t) * length);

	int* compacted_length = NULL;
	cudaMalloc((void**)&compacted_length, sizeof(int));

	compactMask << <numBlocksMasks, threadsPerBlock >> > (scanned_mask, compacted_mask, compacted_length, length);

	uint32_t* host_compacted_mask = (uint32_t*)malloc(sizeof(uint32_t) * length);
	cudaMemcpy(host_compacted_mask, compacted_mask, sizeof(uint32_t) * length, cudaMemcpyDeviceToHost);

	int* host_compacted_length = (int*) malloc(sizeof(int));
	cudaMemcpy(host_compacted_length, compacted_length, sizeof(int), cudaMemcpyDeviceToHost);
	
	int numBlocksChunks = (*host_compacted_length + threadsPerBlock - 1) / threadsPerBlock;
	uint32_t* chunks = NULL;
	cudaMalloc((void**)&chunks, sizeof(uint32_t) * *host_compacted_length);
	calculateChunks <<<numBlocksChunks, threadsPerBlock >> > (compacted_mask, chunks, compacted_length);

	uint32_t* host_chunks = (uint32_t*)malloc(sizeof(uint32_t) * (*host_compacted_length));
	cudaMemcpy(host_chunks, chunks, sizeof(uint32_t) * (*host_compacted_length), cudaMemcpyDeviceToHost);

	uint32_t* positions = NULL;
	cudaMalloc((void**)&positions, sizeof(uint32_t)* *host_compacted_length);
	calculatePosition(chunks, positions, *host_compacted_length);

	uint32_t* host_positions = (uint32_t*)malloc(sizeof(uint32_t) * (*host_compacted_length));
	cudaMemcpy(host_positions, positions, sizeof(uint32_t) * (*host_compacted_length), cudaMemcpyDeviceToHost);

	uint32_t outputLength = thrust::reduce(thrust::device, chunks, chunks + *host_compacted_length, 0);

	unsigned char* outputBytes = NULL;
	cudaMalloc((void**)&outputBytes, sizeof(unsigned char) * outputLength);

	unsigned char* outputCounts = NULL;
	cudaMalloc((void**)&outputCounts, sizeof(unsigned char) * outputLength);

	finalEncoding<<<numBlocksChunks,threadsPerBlock>>>(compacted_mask, positions, compacted_length, dev_data, outputBytes, outputCounts);
	
	unsigned char* hostOutputBytes = (unsigned char*)malloc(sizeof(unsigned char) * outputLength);
	cudaMemcpy(hostOutputBytes, outputBytes,sizeof(unsigned char) * outputLength, cudaMemcpyDeviceToHost);


	unsigned char* hostOutputCounts = (unsigned char*)malloc(sizeof(unsigned char) * outputLength);
	cudaMemcpy(hostOutputCounts, outputCounts, sizeof(unsigned char) * outputLength, cudaMemcpyDeviceToHost);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "GPU Total time: " << duration.count() << " ms" << std::endl;
	//std::cout << "Bytes" << "\t" << "Counts" << "\n";
	/*for (uint32_t i = 0; i < outputLength; i++) {
		printf("%x \t", hostOutputBytes[i]);
		printf("%x \n", hostOutputCounts[i]);
	}*/
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
__host__ void CudaRLDecoding(unsigned char* encodedValues, unsigned char* encodedCounts, int encodedLength, unsigned char** decoded,int* decodedLength)
{
	unsigned char* dev_encodedValues = NULL;
	cudaMalloc((void**)&dev_encodedValues, sizeof(char) * encodedLength);
	cudaMemcpy(dev_encodedValues, encodedValues, sizeof(char) * encodedLength, cudaMemcpyHostToDevice);

	unsigned char* dev_encodedCounts = NULL;
	cudaMalloc((void**)&dev_encodedCounts, sizeof(char) * encodedLength);
	cudaMemcpy(dev_encodedCounts, encodedCounts, sizeof(char) * encodedLength, cudaMemcpyHostToDevice);

	uint32_t* positions = NULL;
	cudaMalloc((void**)&positions, sizeof(unsigned char) * encodedLength);
	calculatePosition(dev_encodedCounts, positions, encodedLength);

	*decodedLength = thrust::reduce(thrust::device, dev_encodedCounts, dev_encodedCounts + encodedLength);

	unsigned char* dev_decoded = NULL;
	cudaMalloc((void**)&dev_decoded, sizeof(char) * *decodedLength);

	int threadsPerBlock = 512;
	int numBlocks = (*decodedLength + threadsPerBlock - 1) / threadsPerBlock;

	finalDecoding<<<numBlocks,threadsPerBlock>>>(dev_encodedValues, positions, encodedLength, dev_decoded, *decodedLength);

	*decoded = (unsigned char*) malloc(sizeof(unsigned char) * *decodedLength);
	cudaMemcpy(*decoded, dev_decoded, sizeof(char) * *decodedLength, cudaMemcpyDeviceToHost);
	for (int i = 0; i < *decodedLength; i++) {
		printf("%d \n",(*decoded)[i]);
	}
}
