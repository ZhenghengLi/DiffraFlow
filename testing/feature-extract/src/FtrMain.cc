#include <iostream>
#include <fstream>
#include <cmath>
#include <H5Cpp.h>
#include <stdio.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>
#include <cuda_runtime.h>

#include "FtrOptMan.hh"
#include "FtrConfig.hh"
#include "ImageFileHDF5R.hh"
#include "ImageFeature.hh"
#include "cudatools.hh"

using namespace diffraflow;
using namespace std;

int main(int argc, char** argv) {
    // process command line parameters
    FtrOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }
    // configure logger
    if (option_man.logconf_file.empty()) {
        log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
        static const log4cxx::LogString logfmt(LOG4CXX_STR("%d [%t] %-5p %c - %m%n"));
        log4cxx::LayoutPtr layout(new log4cxx::PatternLayout(logfmt));
        log4cxx::AppenderPtr appender(new log4cxx::ConsoleAppender(layout));
        log4cxx::Logger::getRootLogger()->addAppender(appender);
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    } else {
        log4cxx::PropertyConfigurator::configure(option_man.logconf_file);
    }
    // parse configuration file
    FtrConfig* config = new FtrConfig();
    if (!option_man.config_file.empty() && !config->load(option_man.config_file.c_str())) {
        cerr << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }
    config->print();
    // select gpu
    bool use_gpu = false;
    if (option_man.gpu_index >= 0) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        cout << "GPU device count: " << deviceCount << endl;
        if (option_man.gpu_index >= deviceCount) {
            cerr << "GPU device index " << option_man.gpu_index << " is out of range [0, " << deviceCount << ")"
                 << endl;
            return 1;
        }
        cudaError_t cudaerr = cudaSetDevice(option_man.gpu_index);
        if (cudaerr == cudaSuccess) {
            use_gpu = true;
            cout << "successfully selected device " << option_man.gpu_index << endl;
            cout << cudatools::get_device_string(option_man.gpu_index) << endl;
        } else {
            cerr << "failed to select device " << option_man.gpu_index << " with error: " << cudaGetErrorString(cudaerr)
                 << endl;
            return 1;
        }
    }
    // allocate memory space
    ImageDataField* image_data_host = nullptr;
    ImageDataField* image_data_device = nullptr;
    ImageFeature* image_feature_host = nullptr;
    ImageFeature* image_feature_device = nullptr;
    double* sum_device = nullptr;
    int* count_device = nullptr;
    if (use_gpu) {
        if (cudaMallocHost(&image_data_host, sizeof(ImageDataField)) != cudaSuccess) {
            cerr << "cudaMallocHost failed for image_data_host." << endl;
            return 1;
        }
        if (cudaMalloc(&image_data_device, sizeof(ImageDataField)) != cudaSuccess) {
            cerr << "cudaMalloc failed for image_data_device." << endl;
            return 1;
        }
        if (cudaMallocHost(&image_feature_host, sizeof(ImageFeature)) != cudaSuccess) {
            cerr << "cudaMallocHost failed for image_feature_host." << endl;
            return 1;
        }
        if (cudaMalloc(&image_feature_device, sizeof(ImageFeature)) != cudaSuccess) {
            cerr << "cudaMalloc failed for image_feature_device." << endl;
            return 1;
        }
        if (cudaMalloc(&sum_device, sizeof(double)) != cudaSuccess) {
            cerr << "cudaMalloc failed for sum_device." << endl;
            return 1;
        }
        if (cudaMalloc(&count_device, sizeof(int)) != cudaSuccess) {
            cerr << "cudaMalloc failed for count_device." << endl;
            return 1;
        }
    } else {
        image_data_host = new ImageDataField();
        image_feature_host = new ImageFeature();
    }
    // open output file
    ostream* output = &cout;
    ofstream* outfile = nullptr;
    if (!option_man.output_file.empty()) {
        outfile = new ofstream(option_man.output_file);
        if (outfile->is_open()) {
            output = outfile;
        } else {
            cerr << "Failed to open output file: " << option_man.output_file << endl;
            return 1;
        }
    }

    // ===== process begin =======================================================================
    ImageFileHDF5R image_file(10, false);
    if (!image_file.open(option_man.data_file.c_str())) {
        cerr << "Failed to open image data file: " << option_man.data_file << endl;
        return 1;
    }
    // cout << "create time: " << image_file.create_time() << endl;

    while (image_file.next_batch()) {
        while (image_file.next_image(*image_data_host)) {
            //
            cout << image_file.current_position() << endl;
        }
    }

    image_file.close();

    // ===== process end =========================================================================

    // clean
    if (outfile != nullptr) {
        outfile->close();
        delete outfile;
    }
    delete config;
    config = nullptr;
    if (use_gpu) {
        cudaFreeHost(image_data_host);
        cudaFreeHost(image_feature_host);
        cudaFree(image_data_device);
        cudaFree(image_feature_device);
        cudaFree(sum_device);
        cudaFree(count_device);
    } else {
        delete image_data_host;
        delete image_feature_host;
    }

    return 0;
}
