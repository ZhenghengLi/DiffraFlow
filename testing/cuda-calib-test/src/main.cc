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

    // prepare calib data
    CalibDataField* calib_data_host = nullptr;
    if (cudaMallocHost(&calib_data_host, sizeof(CalibDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for calib_data_host." << endl;
        return 1;
    }
    for (int m = 0; m < MOD_CNT; m++) {
        for (int l = 0; l < GLV_CNT; l++) {
            for (int h = 0; h < FRAME_H; h++) {
                for (int w = 0; w < FRAME_W; w++) {
                    calib_data_host->pedestal[m][l][h][w] = 1000 + random() % 4000;
                    calib_data_host->gain[m][l][h][w] = 1.0 / (100 + random() % 100);
                }
            }
        }
    }

    // prepare image data
    ImageDataField* image_data_host = nullptr;
    ImageDataField* image_data_host_tmp = nullptr;
    ImageDataField* image_data_host_cpu = nullptr;
    ImageDataField* image_data_host_gpu = nullptr;
    if (cudaMallocHost(&image_data_host, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host." << endl;
        return 1;
    }
    if (cudaMallocHost(&image_data_host_tmp, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host_tmp." << endl;
        return 1;
    }
    if (cudaMallocHost(&image_data_host_cpu, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host_cpu." << endl;
        return 1;
    }
    if (cudaMallocHost(&image_data_host_gpu, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host_gpu." << endl;
        return 1;
    }
    // init
    for (int m = 0; m < MOD_CNT; m++) {
        image_data_host->alignment[m] = true;
        for (int h = 0; h < FRAME_H; h++) {
            for (int w = 0; w < FRAME_W; w++) {
                image_data_host->pixel_data[m][h][w] = 6000 + random() % 9000;
                image_data_host->gain_level[m][h][w] = random() % GLV_CNT;
            }
        }
    }
    image_data_host->calib_level = 0;

    // allocate memory on device
    CalibDataField* calib_data_device = nullptr;
    ImageDataField* image_data_device = nullptr;
    if (cudaMalloc(&calib_data_device, sizeof(CalibDataField)) != cudaSuccess) {
        cerr << "cudaMalloc failed for calib_data_device." << endl;
        return 1;
    }
    if (cudaMalloc(&image_data_device, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMalloc failed for image_data_device." << endl;
        return 1;
    }
    // copy calibration data into device
    if (cudaMemcpy(calib_data_device, calib_data_host, sizeof(CalibDataField), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "cudaMemcpy failed for calibration data." << endl;
        return 1;
    }

    // CPU test begin ///////////////////////////////////////////////
    if (!gpuOnly) {

        cout << endl << "do test on CPU with total " << repeatNum << " images ..." << endl;

        duration<double, micro> start_time = system_clock::now().time_since_epoch();

        for (int r = 0; r < repeatNum; r++) {
            memcpy(image_data_host_tmp, image_data_host, sizeof(ImageDataField));
            Calibration::do_calib_cpu(image_data_host_tmp, calib_data_host);
            memcpy(image_data_host_cpu, image_data_host_tmp, sizeof(ImageDataField));
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
        cudaMemcpyAsync(image_data_device, image_data_host, sizeof(ImageDataField), cudaMemcpyHostToDevice, stream);
        Calibration::do_calib_gpu(stream, image_data_device, calib_data_device);
        cudaMemcpyAsync(image_data_host_gpu, image_data_device, sizeof(ImageDataField), cudaMemcpyDeviceToHost, stream);
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
        cout << "ADC: " << flush;
        for (int i = 0; i < 10; i++) {
            cout << image_data_host->pixel_data[0][0][i] << " ";
        }
        cout << endl;
        cout << "CPU: " << flush;
        for (int i = 0; i < 10; i++) {
            cout << image_data_host_cpu->pixel_data[0][0][i] << " ";
        }
        cout << endl;
        cout << "GPU: " << flush;
        for (int i = 0; i < 10; i++) {
            cout << image_data_host_gpu->pixel_data[0][0][i] << " ";
        }
        cout << endl;

        bool success = true;
        for (int m = 0; m < MOD_CNT; m++) {
            for (int h = 0; h < FRAME_H; h++) {
                for (int w = 0; w < FRAME_W; w++) {
                    if (abs(image_data_host_gpu->pixel_data[m][h][w] - image_data_host_cpu->pixel_data[m][h][w]) >
                        1e-5) {
                        success = false;
                    }
                }
            }
        }
        if (success) {
            cout << endl << "Test PASSED." << endl;
        } else {
            cout << endl << "Test FAILED." << endl;
        }
    }

    // check end ////////////////////////////////////////////////////

    // clean data
    cudaFreeHost(calib_data_host);

    cudaFreeHost(image_data_host);
    cudaFreeHost(image_data_host_tmp);
    cudaFreeHost(image_data_host_cpu);
    cudaFreeHost(image_data_host_gpu);

    cudaFree(calib_data_device);
    cudaFree(image_data_device);

    return 0;
}