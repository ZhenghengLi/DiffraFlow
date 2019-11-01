#ifndef UTIL_SERIALIZE_H
#define UTIL_SERIALIZE_H

#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>

namespace shine {

    size_t serialize_double(double value, char* buffer_p, size_t buffer_l);
    size_t serialize_float(float value, char* buffer_p, size_t buffer_l);
    size_t serialize_int64(int64_t value, char* buffer_p, size_t buffer_l);
    size_t serialize_int32(int32_t value, char* buffer_p, size_t buffer_l);
    size_t serialize_int16(int16_t value, char* buffer_p, size_t buffer_l);

    size_t deserialize_double(double& value, char* buffer_p, size_t buffer_l);
    size_t deserialize_float(float& value, char* buffer_p, size_t buffer_l);
    size_t deserialize_int64(int64_t& value, char* buffer_p, size_t buffer_l);
    size_t deserialize_int32(int32_t& value, char* buffer_p, size_t buffer_l);
    size_t deserialize_int16(int16_t& value, char* buffer_p, size_t buffer_l);

}

#endif