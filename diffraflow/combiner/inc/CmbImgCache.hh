#ifndef CmbImgCache_H
#define CmbImgCache_H

#include "BlockingQueue.hh"
#include "OrderedQueue.hh"
#include "MetricsProvider.hh"

#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <log4cxx/logger.h>

#include <list>

using std::queue;
using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::shared_ptr;
using std::make_shared;
using std::thread;
using std::list;

namespace diffraflow {

    class ImageFrameRaw;
    class ImageDataRaw;

    class CmbImgCache : public MetricsProvider {
    public:
        explicit CmbImgCache(size_t num_of_dets, size_t img_q_ms = 100, int max_lt = 30000);
        ~CmbImgCache();

        void set_distance_threshold(int64_t distance);
        void set_queue_size_threshold(size_t queue_size);

        bool push_frame(const shared_ptr<ImageFrameRaw>& image_frame);
        bool take_image(shared_ptr<ImageDataRaw>& image_data);
        void stop(int wait_time = 0 /* millisecond */);

    public:
        struct {
            atomic<uint64_t> total_pushed_frames;
            atomic<uint64_t> total_aligned_images;
            atomic<uint64_t> total_late_arrived;
            atomic<uint64_t> total_partial_images;
        } alignment_metrics;

        struct {
            atomic<uint64_t> image_data_queue_push_counts;
            atomic<uint64_t> image_data_queue_push_fail_counts;
            atomic<uint64_t> image_data_queue_take_counts;
        } queue_metrics;

        json::value collect_metrics() override;

    private:
        shared_ptr<ImageDataRaw> do_alignment_(bool force_flag = false);
        void clear_cache_();

    private:
        size_t imgfrm_queues_len_;
        list<shared_ptr<ImageFrameRaw>>* imgfrm_queues_arr_;
        BlockingQueue<shared_ptr<ImageDataRaw>> imgdat_queue_;

        mutex data_mtx_;

        atomic_bool stopped_;
        mutex stop_mtx_;
        condition_variable stop_cv_;

        uint64_t key_min_;
        uint64_t key_last_;
        size_t num_of_empty_;
        int64_t distance_max_;
        size_t queue_size_max_;

        int64_t distance_threshold_;
        size_t queue_size_threshold_;

        double latest_push_time_;
        double max_linger_time_; // milliseconds
        bool clear_flag_;
        mutex clear_mtx_;
        condition_variable clear_cv_;
        thread* clear_worker_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
