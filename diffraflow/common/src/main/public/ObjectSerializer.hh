#ifndef ObjectSerializer_H
#define ObjectSerializer_H

#include <cstddef>
#include <cstdint>

// serialized object has this general structure:
// 0 -3 : uint32_t head:0xFEEDBEEF
// 4 -7 : uint32_t size
// 8 -11: int32_t  type
// 12-x : contents

namespace diffraflow {
    class ObjectSerializer {
    public:
        virtual size_t serialize(char* const data, size_t len) = 0;
        virtual size_t deserialize(const char* const data, size_t len) = 0;
        virtual size_t object_size() = 0;
        virtual int object_type() = 0;
        virtual void clear_data() = 0;

    public:
        static const uint32_t kObjectHead = 0xFEEDBEEF;

    };
}

#endif
