#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <chrono>
#include <cstring>

#include <cuda_runtime.h>

#include "ImageDataField.hh"
#include "ImageFeature.hh"
#include "FeatureExtraction.hh"
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

    // select device and print its name on success
    cudaError_t cudaerr = cudaSetDevice(deviceIndex);
    if (cudaerr == cudaSuccess) {
        cout << "successfully selected device " << deviceIndex << endl;
        cout << cudatools::get_device_string(deviceIndex) << endl;
    } else {
        cout << "failed to select device " << deviceIndex << " with error: " << cudaGetErrorString(cudaerr) << endl;
        return 1;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    // prepare image data
    ImageDataField* image_data_host = nullptr;
    if (cudaMallocHost(&image_data_host, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host." << endl;
        return 1;
    }
    ImageFeature* image_feature_host = nullptr;
    if (cudaMallocHost(&image_feature_host, sizeof(ImageFeature)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_feature." << endl;
        return 1;
    }
    ImageFeature* image_feature_host_gpu = nullptr;
    if (cudaMallocHost(&image_feature_host_gpu, sizeof(ImageFeature)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_feature_host_gpu." << endl;
        return 1;
    }
    // init
    unsigned int seed = time(NULL);
    srandom(seed);
    for (int m = 0; m < MOD_CNT; m++) {
        image_data_host->alignment[m] = true;
        for (int h = 0; h < FRAME_H; h++) {
            for (int w = 0; w < FRAME_W; w++) {
                image_data_host->pixel_data[m][h][w] = 500 + random() % 1000;
                image_data_host->gain_level[m][h][w] = random() % GLV_CNT;
            }
        }
    }
    image_data_host->calib_level = 0;
    image_feature_host->clear();

    // allocate memory on device
    ImageDataField* image_data_device = nullptr;
    if (cudaMalloc(&image_data_device, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMalloc failed for image_data_device." << endl;
        return 1;
    }
    ImageFeature* image_feature_device = nullptr;
    if (cudaMalloc(&image_feature_device, sizeof(ImageFeature)) != cudaSuccess) {
        cerr << "cudaMalloc failed for image_feature_device." << endl;
        return 1;
    }
    double* sum_device = nullptr;
    if (cudaMalloc(&sum_device, sizeof(double)) != cudaSuccess) {
        cerr << "cudaMalloc failed for sum_device." << endl;
        return 1;
    }
    int* count_device = nullptr;
    if (cudaMalloc(&count_device, sizeof(int)) != cudaSuccess) {
        cerr << "cudaMalloc failed for count_device." << endl;
        return 1;
    }

    // copy data from cpu to gpu
    if (cudaMemcpy(image_data_device, image_data_host, sizeof(ImageDataField), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "cudaMemcpy failed for image data." << endl;
        return 1;
    }
    if (cudaMemcpy(image_feature_device, image_feature_host, sizeof(ImageFeature), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        cerr << "cudaMemcpy failed for image feature." << endl;
        return 1;
    }

    // CPU test begin ///////////////////////////////////////////////
    if (!gpuOnly) {

        cout << endl << "do test on CPU with total " << repeatNum << " images ..." << endl;

        duration<double, micro> start_time = system_clock::now().time_since_epoch();

        for (int r = 0; r < repeatNum; r++) {
            FeatureExtraction::peak_pixels_MSSE_cpu(image_data_host, image_feature_host, -10, 2000, 1.0, 2.0, 10);
        }

        duration<double, micro> finish_time = system_clock::now().time_since_epoch();
        double time_used = finish_time.count() - start_time.count();

        double cpu_fps = repeatNum * 1000000.0 / time_used;
        cout << "CPU result: " << (long)cpu_fps << " fps" << endl;
    }
    // CPU test end /////////////////////////////////////////////////

    // GPU test begin ///////////////////////////////////////////////
    cout << endl << "do test on GPU with total " << repeatNum << " images ..." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventRecord(start, stream);

    for (int r = 0; r < repeatNum; r++) {
        FeatureExtraction::peak_pixels_MSSE_gpu(
            stream, image_data_device, image_feature_device, -10, 2000, 1.0, 2.0, 10);
        cudaStreamSynchronize(stream);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float msecTotal = 1.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double gpu_fps = repeatNum * 1000.0 / msecTotal;
    cout << "GPU result: " << (long)gpu_fps << " fps" << endl;

    cudaStreamDestroy(stream);
    // GPU test end /////////////////////////////////////////////////

    // check begin //////////////////////////////////////////////////
    if (!gpuOnly) {

        cout << endl << "check results ..." << endl;

        // copy image feature from gpu to cpu
        if (cudaMemcpy(image_feature_host_gpu, image_feature_device, sizeof(ImageFeature), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            cerr << "cudaMemcpy failed for image feature host gpu." << endl;
            return 1;
        }
        cout << "CPU:" << endl;
        cout << "- peak pixels: " << image_feature_host->peak_pixels << endl;
        cout << "GPU:" << endl;
        cout << "- peak pixels: " << image_feature_host_gpu->peak_pixels << endl;
    }

    // check end ////////////////////////////////////////////////////

    // clean data
    cudaFreeHost(image_data_host);
    cudaFreeHost(image_feature_host);
    cudaFree(image_data_device);
    cudaFree(image_feature_device);

    return 0;
}