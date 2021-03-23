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
using std::make_shared;

namespace diffraflow {
    class IngImgDatFetcher : public GenericClient {
    public:
        IngImgDatFetcher(string combiner_host, int combiner_port, uint32_t ingester_id, IngImgWthFtrQueue* queue,
            bool use_gpu = false);
        IngImgDatFetcher(string combiner_sock, uint32_t ingester_id, IngImgWthFtrQueue* queue, bool use_gpu = false);
        ~IngImgDatFetcher();

        void set_recnxn_policy(size_t wait_time, size_t max_count);

        bool start();
        void wait();
        int stop();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };
        enum ReceiveRes { kSucc, kFail, kDisconnected };

    private:
        bool connect_to_combiner_();
        ReceiveRes receive_one_image(shared_ptr<ImageWithFeature>& image_with_feature);

    private:
        int run_();
        shared_future<int> worker_;

        bool use_gpu_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        size_t recnxn_wait_time_;
        size_t recnxn_max_count_;
        size_t max_successive_fail_count_;

        IngImgWthFtrQueue* imgWthFtrQue_raw_;

        mutex cnxn_mtx_;
        condition_variable cnxn_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif