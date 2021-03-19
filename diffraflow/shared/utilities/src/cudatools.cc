#include "cudatools.hh"
#include <sstream>
#include <cuda_runtime.h>

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

string diffraflow::cudatools::get_device_string(int device_index) {
    stringstream ss;
    cudaDeviceProp device_prop;
    cudaError_t cudaerr = cudaGetDeviceProperties(&device_prop, device_index);
    if (cudaerr == cudaSuccess) {
        ss << "GPU " << device_index << ": " << device_prop.name << " (UUID: GPU-"
           << uuid_to_string(device_prop.uuid.bytes) << ")";
    } else {
        ss << "ERROR: " << cudaGetErrorString(cudaerr);
    }
    return ss.str();
}