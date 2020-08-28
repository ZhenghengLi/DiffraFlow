#ifndef __IngFeatureExtracter_H__
#define __IngFeatureExtracter_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>

#include "IngImgWthFtrQueue.hh"

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::async;

namespace diffraflow {
    class IngFeatureExtracter {
    public:
        IngFeatureExtracter(IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out);

        ~IngFeatureExtracter();

        bool start();
        void wait();
        int stop();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        void extract_feature_(const ImageDataType::Field& image_data, ImageFeature& image_feature);

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgWthFtrQueue* image_queue_in_;
        IngImgWthFtrQueue* image_queue_out_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif