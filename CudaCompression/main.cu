﻿#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "run-length_kernels.cuh"
#include "fixed-length_kernels.cuh";
#include "fixed-length_cpu.h";

struct CompressArguments {
    enum Operation { Compress, Decompress, InvalidOperation } operation;
    enum Method { RunLength, FixedLength, InvalidMethod } method;
    std::string inputPath;
    std::string outputPath;

    CompressArguments()
        : operation(InvalidOperation), method(InvalidMethod), inputPath(""), outputPath("") {}
};

CompressArguments readArguments(int argc, char** argv) {
    CompressArguments args;

    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " operation method input_file output_file \n";
        std::cout << "Possible operations: c (compression), d (decompression) \n";
        std::cout << "Possible methods: rl (run-length), fl (fixed-length) \n";
        return args; 
    }

    std::string operation = argv[1];
    if (operation == "c") {
        args.operation = CompressArguments::Compress;
    }
    else if (operation == "d") {
        args.operation = CompressArguments::Decompress;
    }
    else {
        args.operation = CompressArguments::InvalidOperation;
    }

    std::string method = argv[2];
    if (method == "rl") {
        args.method = CompressArguments::RunLength;
    }
    else if (method == "fl") {
        args.method = CompressArguments::FixedLength;
    }
    else {
        args.method = CompressArguments::InvalidMethod;
    }

    args.inputPath = argv[3];
    args.outputPath = argv[4];

    if (args.operation == CompressArguments::InvalidOperation || args.method == CompressArguments::InvalidMethod) {
        std::cout << "Invalid arguments provided. See usage below:\n";
        std::cout << "Usage: " << argv[0] << " operation method input_file output_file \n";
        std::cout << "Possible operations: c (compression), d (decompression) \n";
        std::cout << "Possible methods: rl (run-length), fl (fixed-length) \n";
        return CompressArguments(); 
    }

    return args;
}


RLData readRLCompressedFile(std::string filePath) {
    FILE* file = fopen(filePath.c_str(), "rb");
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return RLData();
    }
    
    RLData compressionData;
    if (fread(&compressionData.length, sizeof(uint64_t), 1, file) != 1) 
    {
        std::cerr << "Failed to read length from file: " << filePath << std::endl;
        fclose(file);
        return RLData();
    }

    compressionData.values.resize(compressionData.length);
    compressionData.counts.resize(compressionData.length);

    if (fread(compressionData.counts.data(), sizeof(char), compressionData.length, file) != compressionData.length)
    {
        std::cerr << "Failed to read counts from file: " << filePath << std::endl;
        fclose(file);
        return RLData();
    }
    if (fread(compressionData.values.data(), sizeof(char), compressionData.length, file) != compressionData.length)
    {
        std::cerr << "Failed to read values from file: " << filePath << std::endl;
        fclose(file);
        return RLData();
    }
    fclose(file);

    return compressionData;
}

bool writeRLCompressedFile(std::string filePath, RLData compressedData) {
    FILE* file = fopen(filePath.c_str(), "wb");
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    if (fwrite(&(compressedData.length), sizeof(uint64_t), 1, file) != 1) {
        std::cerr << "Failed to write length to file: " << filePath << std::endl;
        fclose(file);
        return false;
    }

    if (fwrite(compressedData.counts.data(), sizeof(char), compressedData.length, file) != compressedData.length) {
        std::cerr << "Failed to write counts to file: " << filePath << std::endl;
        fclose(file);
        return false;
    }

    if (fwrite(compressedData.values.data(), sizeof(char), compressedData.length, file) != compressedData.length) {
        std::cerr << "Failed to write values to file: " << filePath << std::endl;
        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}

bool writeDecompressedFile(const std::string filePath, const std::vector<unsigned char> data) {
    FILE* file = fopen(filePath.c_str(), "wb");
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filePath << std::endl;
        return false;
    }

    size_t size = data.size();
    if (fwrite(data.data(), sizeof(unsigned char), size, file) != size) {
        std::cerr << "Failed to write data to file: " << filePath << std::endl;
        fclose(file);
        return false;
    }

    fclose(file);
    return true;
}

std::vector<unsigned char> readUncompressedFile(std::string filePath) {
    FILE* file = fopen(filePath.c_str(), "rb");
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        std::cerr << "Failed to seek to end of file: " << filePath << std::endl;
        fclose(file);
        return {};
    }

    long size = ftell(file);
    if (size == -1) {
        std::cerr << "Failed to get file size: " << filePath << std::endl;
        fclose(file);
        return {};
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        std::cerr << "Failed to seek to beginning of file: " << filePath << std::endl;
        fclose(file);
        return {};
    }
    std::vector<unsigned char> data(size);
    size_t s;
    if ((s = fread(data.data(),sizeof(char),size,file)) != size) {
        std::cerr << "Failed to read file: " << filePath << " " << s << " " << size << std::endl;
        fclose(file);
        return {};
    }
    fclose(file);
    return data;
}


bool CompressFile(CompressArguments args)
{
    std::vector<unsigned char> fileBytes = readUncompressedFile(args.inputPath);
    if (fileBytes.empty())
        return false;
    if (args.method == CompressArguments::RunLength)
    {
        RLData data = CudaRLEncode(fileBytes);
        if (data.length == 0)
            return false;
        return writeRLCompressedFile(args.outputPath, data);
    }
    else {
        return false;
    }
}

bool DecompressFile(CompressArguments args) {
    if (args.method == CompressArguments::RunLength)
    {
        RLData encodedData = readRLCompressedFile(args.inputPath);
        if (encodedData.length == 0)
            return false;
        std::vector<unsigned char> decodedData = CudaRLDecode(encodedData);
        if (decodedData.size() == 0)
            return false;
        return writeDecompressedFile(args.outputPath, decodedData);
    }
    else {
        return false;
    }
}
int main(int argc,char** argv)
{
	CompressArguments args = readArguments(argc,argv);
    if (args.operation == CompressArguments::InvalidOperation ||
        args.method == CompressArguments::InvalidMethod) {
        return EXIT_FAILURE; 
    }
    if (args.operation == CompressArguments::Compress)
    {
        if (!CompressFile(args))
            return EXIT_FAILURE;
    }
    else {
        if (!DecompressFile(args))
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

