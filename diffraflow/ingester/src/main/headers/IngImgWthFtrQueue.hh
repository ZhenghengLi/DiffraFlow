#ifndef __IngImgWthFtrQueue_H__
#define __IngImgWthFtrQueue_H__

#include "BlockingQueue.hh"
#include "ImageWithFeature.hh"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class IngImgWthFtrQueue {
    public:
        explicit IngImgWthFtrQueue(size_t img_q_ms = 100);
        ~IngImgWthFtrQueue();

        bool push(const shared_ptr<ImageWithFeature>& image_with_feature);
        bool take(shared_ptr<ImageWithFeature>& image_with_feature);
        void stop(int wait_time = 0 /* millisecond */);

    private:
        BlockingQueue<shared_ptr<ImageWithFeature>> imgWthFtr_queue_;

        atomic_bool stopped_;
        mutex stop_mtx_;
        condition_variable stop_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif