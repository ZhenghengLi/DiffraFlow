#ifndef DspSender_H
#define DspSender_H

#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>

#include "GenericClient.hh"

using std::string;
using std::thread;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;

namespace diffraflow {
    class DspSender: public GenericClient {
    public:
        DspSender(string hostname, int port, int id, bool compr_flag = false);
        ~DspSender();

        // push to buffer_A and block on buffer full
        void push(const char* data, const size_t len);
        void send_remaining();

        // use a background thread sending data
        void start();
        void stop();

    private:
        // swap buffer_A and buffer_B with lock
        bool swap_();
        // send buffer_B over TCP
        void send_();
        // help function
        void send_buffer_(const char* buffer, const size_t limit, const size_t num_of_imgs);

    private:
        // buffer
        size_t buffer_size_;
        char*  buffer_A_;
        size_t buffer_A_limit_;
        size_t buffer_A_imgct_;
        char*  buffer_B_;
        size_t buffer_B_limit_;
        size_t buffer_B_imgct_;
        size_t size_threshold_;
        size_t time_threshold_; // ms
        // sending thread
        thread* sending_thread_;
        mutex mtx_;
        mutex mtx_send_;
        condition_variable cv_push_;
        condition_variable cv_swap_;
        atomic_bool run_flag_;
        bool compress_flag_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
