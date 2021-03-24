#ifndef __IngCalibrationWorker_H__
#define __IngCalibrationWorker_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <log4cxx/logger.h>
#include <cuda_runtime.h>

#include "CalibDataField.hh"
#include "ImageDimension.hh"
#include "IngBufferItemQueue.hh"

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::shared_ptr;
using std::async;

namespace diffraflow {

    class IngImgFtrBuffer;

    class IngCalibrationWorker {
    public:
        IngCalibrationWorker(
            IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngBufferItemQueue* queue_out, bool use_gpu = false);

        ~IngCalibrationWorker();

        bool start();
        void wait();
        int stop();

        bool read_calib_file(const char* calib_file);

    private:
        CalibDataField* calib_data_host_;
        CalibDataField* calib_data_device_;

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        void do_calib_(const IngBufferItem& item);

    private:
        int run_();
        shared_future<int> worker_;

        bool use_gpu_;
        cudaStream_t cuda_stream_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgFtrBuffer* image_feature_buffer_;
        IngBufferItemQueue* item_queue_in_;
        IngBufferItemQueue* item_queue_out_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif