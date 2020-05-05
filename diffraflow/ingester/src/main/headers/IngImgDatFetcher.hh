#ifndef __IngImgDatFetcher_H__
#define __IngImgDatFetcher_H__

#include "GenericClient.hh"
#include "IngImgWthFtrQueue.hh"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::future;
using std::shared_future;
using std::async;

namespace diffraflow {
    class IngImgDatFetcher : private GenericClient {
    public:
        IngImgDatFetcher(string combiner_host, int combiner_port, uint32_t ingester_id, IngImgWthFtrQueue* queue);
        ~IngImgDatFetcher();

        void set_recnxn_policy(size_t wait_time, size_t max_count);

        bool start();
        void wait();
        int stop();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };
        enum RequestRes { kSucc, kFail, kDisconnected };

    private:
        bool connect_to_combiner_();
        RequestRes request_one_image(ImageData& image_data);

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        size_t recnxn_wait_time_;
        size_t recnxn_max_count_;
        size_t max_successive_fail_count_;

        IngImgWthFtrQueue* imgWthFtrQue_raw_;

        char* imgdat_buffer_;
        size_t imgdat_buffer_size_;

        mutex cnxn_mtx_;
        condition_variable cnxn_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif