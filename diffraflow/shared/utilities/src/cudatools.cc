#include "cudatools.hh"
#include <sstream>

using std::string;
using std::stringstream;

string diffraflow::cudatools::uuid_to_string(char uuid_bytes[16]) {
    stringstream ss;
    ss << std::hex;
    for (int i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) ss << "-";
        ss << (int)((uint8_t*)uuid_bytes)[i];
    }
    return ss.str();
}