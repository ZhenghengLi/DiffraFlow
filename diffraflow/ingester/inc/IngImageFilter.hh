#ifndef __IngImageFilter_H__
#define __IngImageFilter_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>
#include <cuda_runtime.h>

#include "MetricsProvider.hh"
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
    class IngConfig;
    class ImageFeature;

    class IngImageFilter : public MetricsProvider {
    public:
        IngImageFilter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngBufferItemQueue* queue_out,
            IngConfig* conf_obj, bool use_gpu = false, int gpu_index = 0);

        ~IngImageFilter();

        bool start();
        void wait();
        int stop();

    public:
        struct {
            atomic<uint64_t> total_processed_images;
            atomic<uint64_t> total_images_for_save;
            atomic<uint64_t> total_images_for_save_fail;
            atomic<uint64_t> total_images_for_monitor;
        } filter_metrics;

        json::value collect_metrics() override;

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        void do_filter(shared_ptr<IngBufferItem>& item);
        bool check_for_save_(const ImageFeature* image_feature);
        bool check_for_monitor_(const ImageFeature* image_feature);

    private:
        int run_();
        shared_future<int> worker_;

        bool use_gpu_;
        int gpu_index_;
        cudaStream_t cuda_stream_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgFtrBuffer* image_feature_buffer_;
        IngBufferItemQueue* item_queue_in_;
        IngBufferItemQueue* item_queue_out_;
        IngConfig* config_obj_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif