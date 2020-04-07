#ifndef __IngImageFilter_H__
#define __IngImageFilter_H__

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

    class IngImageFilter {
    public:
        IngImageFilter(
            IngImgWthFtrQueue*  img_queue_in,
            IngImgWthFtrQueue*  img_queue_out,
            IngConfig*          conf_obj);

        ~IngImageFilter();

        bool start();
        void wait();
        int  stop();

        void set_current_image(const shared_ptr<ImageWithFeature>& image_with_feature);
        ImageWithFeature get_current_image();

    public:
        enum WorkerStatus {kNotStart, kRunning, kStopped};

    private:
        bool check_for_save_(const ImageFeature& image_feature);
        bool check_for_monitor_(const ImageFeature& image_feature);

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgWthFtrQueue*  image_queue_in_;
        IngImgWthFtrQueue*  image_queue_out_;
        IngConfig*          config_obj_;

    private:
        shared_ptr<ImageWithFeature>    current_image_;
        mutex                           current_image_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif