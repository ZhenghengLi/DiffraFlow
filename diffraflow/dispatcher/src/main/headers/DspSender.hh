#ifndef DspSender_H
#define DspSender_H

#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <atomic>

using std::string;
using std::thread;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;

namespace diffraflow {
    class DspSender {
    public:
        DspSender(string hostname, int port, int id);
        ~DspSender();

        bool connect_to_combiner();
        void close_connection();

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
        void send_buffer_(const char* buffer, char* buff_compr,
            const size_t limit, const size_t num_of_imgs);

    private:
        // socket
        string dest_host_;
        int    dest_port_;
        int    sender_id_;
        int    client_sock_fd_;
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
        // buffer for compression
        char*  buff_compr_A_;
        char*  buff_compr_B_;
        // sending thread
        thread* sending_thread_;
        mutex mtx_;
        condition_variable cv_push_;
        condition_variable cv_swap_;
        atomic_bool run_flag_;

    };
}

#endif
