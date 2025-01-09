#pragma once
#include <stdio.h>
#include <iostream>
#include "run-length_kernels.cuh"
#include "fixed-length_kernels.cuh";

int main(int argc,char** argv)
{
	/*if (argc != 4)
	{
		std::cout << "Usage: " << argv[0] << " <input_file> <output_compressed_file> <output_diff_file>" << std::endl;
		return EXIT_FAILURE;
	}
	
	FILE* input_file = fopen(argv[1], "r");
	if (input_file == NULL) {
		std::cerr << "Error reading input_file" << std::endl;
		return EXIT_FAILURE;
	}*/
	/*int c = 5900000;
	unsigned char* data = (unsigned char*) malloc(sizeof(char) * c);
	int length = 50;

	for (int i = 0; i < c; i++) {
		data[i] = 7;
	}
	CudaRLEncoding(data, c);*/
	//unsigned char encodedValues[] = { 1,2,3 };       // Example values
	//unsigned char encodedCounts[] = { 3, 0, 5 };             // Example counts
	//int encodedLength = sizeof(encodedValues) / sizeof(encodedValues[0]);

	//// Output variables
	//unsigned char* decoded = nullptr;
	//int decodedLength = 0;

	//// Call the CudaRLDecoding function
	//CudaRLDecoding(encodedValues, encodedCounts, encodedLength, &decoded, &decodedLength);
	unsigned char data[] = {3, 255, 2, 1, 0, 6, 2, 1};
	int length = 8;
	CudaFLEncoding(data, length);
}

