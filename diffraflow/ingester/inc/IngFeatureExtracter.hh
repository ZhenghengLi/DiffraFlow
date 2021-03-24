#ifndef __IngFeatureExtracter_H__
#define __IngFeatureExtracter_H__

#include <mutex>
#include <atomic>
#include <memory>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>

#include "IngBufferItemQueue.hh"

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::shared_ptr;
using std::async;
using std::make_shared;

namespace diffraflow {

    class IngImgFtrBuffer;

    class IngFeatureExtracter {
    public:
        IngFeatureExtracter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngBufferItemQueue* queue_out);

        ~IngFeatureExtracter();

        bool start();
        void wait();
        int stop();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        void extract_feature_(const IngBufferItem& item);

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        IngImgFtrBuffer* image_feature_buffer_;
        IngBufferItemQueue* item_queue_in_;
        IngBufferItemQueue* item_queue_out_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif