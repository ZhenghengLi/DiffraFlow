#ifndef __IngImageWriter_H__
#define __IngImageWriter_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::shared_ptr;
using std::async;

namespace diffraflow {

    class IngImgWthFtrQueue;
    class ImageWithFeature;
    class ImageFeature;
    class IngConfig;

    class IngImageWriter {
    public:
        IngImageWriter(
            IngImgWthFtrQueue*  img_queue_in,
            IngConfig*          conf_obj);

        ~IngImageWriter();

        bool start();
        void wait();
        int  stop();

    public:
        enum WorkerStatus {kNotStart, kRunning, kStopped};

    private:
        bool save_image_(const shared_ptr<ImageWithFeature> & image_with_feature);

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgWthFtrQueue*  image_queue_in_;
        IngConfig*          config_obj_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif