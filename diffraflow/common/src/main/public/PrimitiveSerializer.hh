#ifndef PrimitiveSerializer_H
#define PrimitiveSerializer_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <iostream>

namespace shine {
    class PrimitiveSerializer {
    private:
        bool isLittleEndian_;
    public:
        PrimitiveSerializer();

        template <class T>
        size_t serialize(T value, char* buffer_p, size_t buffer_l);

        template <class T>
        size_t deserialize(T* value_p, char* buffer_p, size_t buffer_l);

    };
}

template <class T>
size_t shine::PrimitiveSerializer::serialize(T value, char* buffer_p, size_t buffer_l) {
    if (!std::is_fundamental<T>::value) {
        std::cerr << "WARNING: cannnot serialize data of non-fundamental types." << std::endl;
        return 0;
    }
    size_t byte_l = sizeof(T);
    if (byte_l < buffer_l) return 0;
    char* const byte_p = reinterpret_cast<char*>(&value);
    if (isLittleEndian_) {
        for (size_t i = 0; i < byte_l; i++) {
            buffer_p[i] = byte_p[byte_l - i - 1];
        }
    } else {
        for (size_t i = 0; i < byte_l; i++) {
            buffer_p[i] = byte_p[i];
        }
    }
    return byte_l;
}

template <class T>
size_t shine::PrimitiveSerializer::deserialize(T* value_p, char* buffer_p, size_t buffer_l) {
    if (!std::is_fundamental<T>::value) {
        std::cerr << "WARNING: cannnot deserialize data of non-fundamental types." << std::endl;
        return 0;
    }
    size_t byte_l = sizeof(T);
    if (byte_l < buffer_l) return 0;
    char* byte_p = reinterpret_cast<char*>(value_p);
    if (isLittleEndian_) {
        for (size_t i = 0; i < byte_l; i++) {
            byte_p[byte_l - i - 1] = buffer_p[i];
        }
    } else {
        for (size_t i = 0; i < byte_l; i++) {
            byte_p[i] = buffer_p[i];
        }
    }
    return byte_l;
}

#endif