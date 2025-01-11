#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
int countLeadingZeros2(unsigned char data);
void calculateFrameBits(const unsigned char* data, size_t length, std::vector<uint32_t>& frameBits);
void calculateFramePositions(const std::vector<uint32_t>& frameBits, std::vector<int>& framePositions);
size_t calculateOutputSize(const std::vector<int>& framePositions,
    const std::vector<uint32_t>& frameBits,
    size_t dataLength);
void fillOutput(const unsigned char* data,
    size_t length,
    const std::vector<uint32_t>& frameBits,
    std::vector<uint32_t>& output,
    const std::vector<int>& framePositions);
std::vector<uint32_t> CPUFixedLengthEncode(unsigned char* data, size_t length);