#include "PrimitiveSerializer.hh"

shine::PrimitiveSerializer::PrimitiveSerializer() {
    short number = 0x1;
    char* num_p = (char*)&number;
    if (num_p[0] == 1) {
        isLittleEndian_ = true;
    } else {
        isLittleEndian_ = false;
    }
}
