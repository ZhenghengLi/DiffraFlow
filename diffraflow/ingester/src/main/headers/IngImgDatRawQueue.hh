#ifndef __IngImgDatRawQueue_H__
#define __IngImgDatRawQueue_H__

#include "BlockingQueue.hh"
#include "ImageData.hh"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::mutex;
using std::condition_variable;
using std::atomic_bool;

namespace diffraflow {
    class IngImgDatRawQueue {
    public:
        explicit IngImgDatRawQueue(size_t img_q_ms = 100);
        ~IngImgDatRawQueue();

        bool push(const ImageData& image_data);
        bool take(ImageData& image_data);
        void stop(int wait_time = 0  /* millisecond */);

    private:
        BlockingQueue<ImageData> imgdat_queue_;

        atomic_bool stopped_;
        mutex stop_mtx_;
        condition_variable stop_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif