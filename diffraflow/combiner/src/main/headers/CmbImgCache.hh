#ifndef CmbImgCache_H
#define CmbImgCache_H

#include "BlockingQueue.hh"
#include "TimeOrderedQueue.hh"
#include "MetricsProvider.hh"

#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::queue;
using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::shared_ptr;
using std::make_shared;
using std::thread;

namespace diffraflow {

    class ImageFramePtr;
    class ImageData;

    class CmbImgCache : public MetricsProvider {
    public:
        explicit CmbImgCache(size_t num_of_dets, size_t img_q_ms = 100, int max_lt = 30000);
        ~CmbImgCache();

        bool push_frame(const ImageFramePtr& image_frame);
        bool take_image(shared_ptr<ImageData>& image_data);
        void stop(int wait_time = 0 /* millisecond */);

    public:
        struct {
            atomic<uint64_t> total_pushed_frames;
            atomic<uint64_t> total_aligned_images;
            atomic<uint64_t> total_late_arrived;
            atomic<uint64_t> total_partial_images;
        } alignment_metrics;

        json::value collect_metrics() override;

    private:
        bool do_alignment_(shared_ptr<ImageData> image_data, bool force_flag = false);
        void clear_cache_();

    private:
        size_t imgfrm_queues_len_;
        TimeOrderedQueue<ImageFramePtr, int64_t>* imgfrm_queues_arr_;
        BlockingQueue<shared_ptr<ImageData>> imgdat_queue_;

        mutex data_mtx_;

        atomic_bool stopped_;
        mutex stop_mtx_;
        condition_variable stop_cv_;

        int64_t wait_threshold_; // nanoseconds
        uint64_t image_time_min_;
        uint64_t image_time_last_;
        size_t num_of_empty_;
        int64_t distance_max_;

        double latest_push_time_;
        double max_linger_time_; // milliseconds
        bool clear_flag_;
        condition_variable clear_cv_;
        thread* clear_worker_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
