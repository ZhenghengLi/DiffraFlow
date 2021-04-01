#ifndef DspSender_H
#define DspSender_H

#include <string>
#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>
#include <sched.h>

#include "GenericClient.hh"
#include "BlockingQueue.hh"

using std::string;
using std::thread;
using std::mutex;
using std::shared_ptr;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;
using std::atomic;
using std::atomic_bool;

namespace diffraflow {

    class ImageFrameRaw;

    class DspSender : public GenericClient {
    public:
        DspSender(string hostname, int port, int id, size_t max_qs = 1000);
        ~DspSender();

        bool push(const shared_ptr<ImageFrameRaw>& image_frame);

        bool start(cpu_set_t* cpuset = nullptr);
        void stop();

    public:
        struct {
            atomic<uint64_t> total_pushed_counts;
            atomic<uint64_t> total_send_succ_counts;
            atomic<uint64_t> total_send_fail_counts;
        } sender_metrics;

        json::value collect_metrics() override;

    private:
        bool send_imgfrm_(const shared_ptr<ImageFrameRaw>& image_frame);

    private:
        BlockingQueue<shared_ptr<ImageFrameRaw>> imgfrm_queue_;
        thread* sending_thread_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
