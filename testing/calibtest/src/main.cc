#include <iostream>
#include <cstdlib>
#include <memory>
#include <chrono>
#include <cstring>

#include <cuda_runtime.h>

#include "ImageDataField.hh"
#include "CalibDataField.hh"
#include "Calibration.hh"
#include "cudatools.hh"

using namespace std;
using namespace diffraflow;

using std::chrono::duration;
using std::chrono::system_clock;
using std::micro;

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "device count: " << deviceCount << endl;

    int deviceIndex = 0;
    int repeatNum = 1;
    int gpuOnly = 0;

    if (argc > 1) {
        repeatNum = atoi(argv[1]);
    }
    if (argc > 2) {
        deviceIndex = atoi(argv[2]);
    }
    if (argc > 3) {
        gpuOnly = atoi(argv[3]);
    }

    if (deviceIndex >= deviceCount) {
        cout << "device index " << deviceIndex << " is out of range [0, " << deviceCount << ")" << endl;
        return 1;
    }

    if (repeatNum < 1) {
        repeatNum = 1;
    }

    cudaError_t cudaerr = cudaSetDevice(deviceIndex);
    if (cudaerr == cudaSuccess) {
        cout << "successfully selected device " << deviceIndex << endl;
        cout << cudatools::get_device_string(deviceIndex) << endl;
    } else {
        cout << "failed to select device " << deviceIndex << " with error: " << cudaGetErrorString(cudaerr) << endl;
        return 1;
    }

    return 0;
}