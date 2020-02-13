#ifndef CmbImgCache_H
#define CmbImgCache_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include "BlockingQueue.hh"
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
        bool do_alignment_();

    private:
        // TODO: use priority_queue instead
        size_t              imgfrm_queues_len_;
        queue<ImageFrame>*  imgfrm_queues_arr_;

        mutex mtx_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
