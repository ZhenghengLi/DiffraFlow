#ifndef DECODER_H
#define DECODER_H

#include <iostream>
#include <cstddef>
#include <cassert>
#include <iomanip>
#include <stdint.h>

namespace diffraflow {
    class Decoder {
    public:
        template <class T>
        T decode_bit(const char* buffer, size_t begin, size_t end);
        template <class T>
        T decode_byte(const char* buffer, size_t begin, size_t end); 
    };

    static Decoder gDC;

    template <class T>
    T Decoder::decode_bit(const char* buffer, size_t begin, size_t end) {
        T sum = 0;
        size_t begin_byte = begin / 8;
        size_t begin_pos = begin % 8;
        size_t end_byte = end / 8;
        size_t end_pos = end % 8;
        sum = static_cast<uint8_t>(buffer[begin_byte]) & ((1 << (8 - begin_pos)) - 1);
        if (begin_byte == end_byte) {
            sum >>= (8 - end_pos - 1);
            return sum;
        } else {
            for (size_t i = 1; begin_byte + i < end_byte; i++) {
                sum <<= 4;
                sum <<= 4; // equivalent to sum <<= 8, when the type of sum is uint8_t this may cause warnings.
                sum += static_cast<uint8_t>(buffer[begin_byte + i]);
            }
            sum <<= end_pos;
            sum <<= 1; // equivalent to sum <<= (end_pos + 1)
            sum += static_cast<uint8_t>(buffer[end_byte]) >> (8 - end_pos -1);
            return sum;
        }
    }

    template <class T>
    T Decoder::decode_byte(const char* buffer, size_t begin, size_t end) {
        T sum = 0;
        for (size_t i = 0; begin + i <= end; i++) {
            sum <<= 4;
            sum <<= 4; // equivalent to sum <<= 8
            sum += static_cast<uint8_t>(buffer[begin + i]);
        }
        return sum;
    }
}

#endif
