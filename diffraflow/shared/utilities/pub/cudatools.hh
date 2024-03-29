#ifndef __cudatools_h__
#define __cudatools_h__

#include <string>

namespace diffraflow {
    namespace cudatools {

        std::string uuid_to_string(char uuid_bytes[16]);
        std::string get_device_string(int device_index);

    } // namespace cudatools
} // namespace diffraflow

#endif
