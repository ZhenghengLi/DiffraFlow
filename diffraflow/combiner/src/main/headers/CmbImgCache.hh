#ifndef CmbImgCache_H
#define CmbImgCache_H

#include "BlockingQueue.hh"
#include "TimeOrderedQueue.hh"

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::queue;
using std::mutex;
using std::condition_variable;
using std::atomic_bool;

namespace diffraflow {

    class ImageFrame;
    class ImageData;

    class CmbImgCache {
    public:
        explicit CmbImgCache(size_t num_of_dets, size_t img_q_ms = 100);
        ~CmbImgCache();

        bool push_frame(const ImageFrame& image_frame);
        bool take_image(ImageData& image_data);
        void stop(int wait_time = 0  /* millisecond */);

    private:
        bool do_alignment_(ImageData& image_data, bool force_flag = false);

    private:
        size_t                                   imgfrm_queues_len_;
        TimeOrderedQueue<ImageFrame, uint64_t>*  imgfrm_queues_arr_;
        BlockingQueue<ImageData>                 imgdat_queue_;
        BlockingQueue<ImageData>                 imgdat_queue_late_;

        mutex data_mtx_;

        atomic_bool stopped_;
        mutex stop_mtx_;
        condition_variable stop_cv_;

        uint64_t wait_threshold_;   // nanoseconds
        uint64_t image_time_min_;
        uint64_t image_time_last_;
        size_t   num_of_empty_;
        uint64_t distance_max_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
