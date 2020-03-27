#ifndef CmbImgCache_H
#define CmbImgCache_H

#include "BlockingQueue.hh"
#include "TimeOrderedQueue.hh"

#include <mutex>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::queue;
using std::mutex;
using std::lock_guard;

namespace diffraflow {

    class ImageFrame;
    class ImageData;

    class CmbImgCache {
    public:
        explicit CmbImgCache(size_t num_of_dets, size_t img_q_ms = 100);
        ~CmbImgCache();

        void push_frame(const ImageFrame& image_frame);

    public:
        BlockingQueue<ImageData> imgdat_queue;

    private:
        bool do_alignment_(bool force_flag = false);

    private:
        size_t                          imgfrm_queues_len_;
        TimeOrderedQueue<ImageFrame>*   imgfrm_queues_arr_;

        mutex mtx_;

        uint64_t wait_threshold_;   // nanoseconds
        uint64_t image_time_min_;
        size_t   num_of_empty_;
        uint64_t distance_max_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
