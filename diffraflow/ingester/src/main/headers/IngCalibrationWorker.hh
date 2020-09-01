#ifndef __IngCalibrationWorker_H__
#define __IngCalibrationWorker_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <log4cxx/logger.h>

#include "IngImgWthFtrQueue.hh"

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::shared_ptr;
using std::async;

namespace diffraflow {
    class IngCalibrationWorker {
    public:
        IngCalibrationWorker(IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out);

        ~IngCalibrationWorker();

        bool start();
        void wait();
        int stop();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        void do_calib_(shared_ptr<ImageWithFeature>& image_with_feature);

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