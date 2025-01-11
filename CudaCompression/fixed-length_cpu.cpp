#include "fixed-length_cpu.h";
#include "fixed-length_kernels.cuh"
const int FRAME_LENGTH = 128;
const int BITS_IN_BYTE = 8;

int countLeadingZeros2(unsigned char data) {
    if (data == 0)
        return 7;
    int count = 0;
    while ((data & (1 << 7)) == 0) {
        count++;
        data <<= 1;
    }
    return count;
}

void calculateFrameBits(const unsigned char* data, size_t length, std::vector<uint32_t>& frameBits) {
    size_t numFrames = (length + FRAME_LENGTH - 1) / FRAME_LENGTH;
    frameBits.resize(numFrames);

    for (size_t frameIndex = 0; frameIndex < numFrames; frameIndex++) {
        int maxBits = 0;
        size_t frameStart = frameIndex * FRAME_LENGTH;
        size_t frameEnd = std::min(frameStart + FRAME_LENGTH, length);

        for (size_t i = frameStart; i < frameEnd; i++) {
            int bits = BITS_IN_BYTE - countLeadingZeros2(data[i]);
            maxBits = std::max(maxBits, bits);
        }
        frameBits[frameIndex] = maxBits;
    }
}

void calculateFramePositions(const std::vector<uint32_t>& frameBits, std::vector<int>& framePositions) {
    framePositions.resize(frameBits.size());
    framePositions[0] = 0;

    for (size_t i = 1; i < frameBits.size(); i++) {
        framePositions[i] = framePositions[i - 1] + frameBits[i - 1] * FRAME_LENGTH;
    }
}



size_t calculateOutputSize(const std::vector<int>& framePositions,
    const std::vector<uint32_t>& frameBits,
    size_t dataLength) {
    if (framePositions.empty() || frameBits.empty()) return 0;

    int lastFrameStart = framePositions.back();
    int lastFrameBitLength = frameBits.back();
    int lastFrameDataCount = dataLength % FRAME_LENGTH == 0 ?
        FRAME_LENGTH : dataLength % FRAME_LENGTH;
    int lastBitPosition = lastFrameStart + lastFrameBitLength * lastFrameDataCount;
    return (lastBitPosition + 31) / 32;
}

void fillOutput(const unsigned char* data,
    size_t length,
    const std::vector<uint32_t>& frameBits,
    std::vector<uint32_t>& output,
    const std::vector<int>& framePositions) {

    for (size_t i = 0; i < length; i++) {
        size_t frameIndex = i / FRAME_LENGTH;
        uint32_t bitsPerSymbol = frameBits[frameIndex];
        size_t symbolIndex = i % FRAME_LENGTH;

        int startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;
        int outputIndex = startPos / 32;
        int bitOffset = startPos % 32;

        unsigned char symbol = data[i];
        uint32_t maskedSymbol = (uint32_t)(symbol & ((1u << bitsPerSymbol) - 1));
        uint32_t shiftedSymbol = maskedSymbol << bitOffset;

        output[outputIndex] |= shiftedSymbol;

        if (bitOffset + bitsPerSymbol > 32) {
            uint32_t spillBits = bitOffset + bitsPerSymbol - 32;
            uint32_t spillMask = maskedSymbol >> (bitsPerSymbol - spillBits);
            output[outputIndex + 1] |= spillMask;
        }
    }
}

void CPUFixedLengthDecode(
    const uint32_t* encodedData,
    int encodedLength,
    const uint32_t* frameBits,
    int frameBitsLength,
    uint64_t decodedDataLength, unsigned char* originalData) {

    std::vector<int> framePositions(frameBitsLength);
    calculateFramePositions(std::vector<uint32_t>(frameBits, frameBits + frameBitsLength),
        framePositions);
    unsigned char* decoded = (unsigned char*)malloc(sizeof(char) * decodedDataLength);
    for (size_t threadId = 0; threadId < decodedDataLength; threadId++) {
        int frameInBlock = threadId / FRAME_LENGTH;
        int frameIndex = frameInBlock; 
        int bitsPerSymbol = frameBits[frameIndex];
        int symbolIndex = threadId % FRAME_LENGTH;

        int startPos = framePositions[frameIndex] + symbolIndex * bitsPerSymbol;
        int outputIndex = startPos / 32;
        int bitOffset = startPos % 32;

        uint32_t data = encodedData[outputIndex];
        data >>= bitOffset;
        data &= (1u << bitsPerSymbol) - 1;

        if (bitOffset + bitsPerSymbol > 32) {
            const int spilledBits = bitOffset + bitsPerSymbol - 32;
            const uint32_t nextWord = encodedData[outputIndex + 1];
            const uint32_t spilledMask = (1u << spilledBits) - 1;
            const uint32_t spilledData = (nextWord & spilledMask) << (bitsPerSymbol - spilledBits);
            data |= spilledData;
        }

        decoded[threadId] = static_cast<unsigned char>(data);
    }
}
std::vector<uint32_t> CPUFixedLengthEncode(unsigned char* data, size_t length) {
    if (length == 0) return {};

    std::vector<uint32_t> frameBits;
    calculateFrameBits(data, length, frameBits);

    std::vector<int> framePositions;
    calculateFramePositions(frameBits, framePositions);

    size_t outputSize = calculateOutputSize(framePositions, frameBits, length);
    std::vector<uint32_t> output(outputSize, 0);  

    fillOutput(data, length, frameBits, output, framePositions);

    return output;
}



