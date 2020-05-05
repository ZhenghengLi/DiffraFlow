#ifndef PrimitiveSerializer_H
#define PrimitiveSerializer_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <stdexcept>

namespace diffraflow {
    class PrimitiveSerializer {
    private:
        bool isLittleEndian_;

    public:
        PrimitiveSerializer();

        template <class T>
        size_t serializeValue(T value, char* const buffer_p, size_t buffer_l);

        template <class T>
        size_t deserializeValue(T* value_p, const char* const buffer_p, size_t buffer_l);
    };

    static PrimitiveSerializer gPS;

} // namespace diffraflow

template <class T>
size_t diffraflow::PrimitiveSerializer::serializeValue(T value, char* const buffer_p, size_t buffer_l) {
    if (!std::is_fundamental<T>::value) {
        throw std::invalid_argument("cannnot serialize data of non-fundamental types.");
    }
    size_t byte_l = sizeof(T);
    if (byte_l > buffer_l) {
        throw std::out_of_range("there is no enough space in buffer to hold the value data.");
    }
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
size_t diffraflow::PrimitiveSerializer::deserializeValue(T* value_p, const char* const buffer_p, size_t buffer_l) {
    if (!std::is_fundamental<T>::value) {
        throw std::invalid_argument("cannnot deserialize data of non-fundamental types.");
    }
    size_t byte_l = sizeof(T);
    if (byte_l > buffer_l) {
        throw std::out_of_range("length of the buffer is shorter than that of the value type.");
    }
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
