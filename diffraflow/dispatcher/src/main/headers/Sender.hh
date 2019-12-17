#ifndef Sender_H
#define Sender_H

#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>

using std::string;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;

namespace diffraflow {
    class Sender {
    public:
        Sender(string hostname, int port, int id);
        ~Sender();

        bool conn();

        // push and block on buffer full
        void push(long key, char* data, size_t len);
        // swap with lock
        bool swap();

        // use a background thread sending data
        void start();
        void stop();

    private:
        // swap and send
        void run_();
        // send over TCP
        void send_();

    private:
        // socket
        string dest_host_;
        int    dest_port_;
        int    sender_id_;
        int    client_sock_fd_;
        // buffer
        size_t buffer_size_;
        char*  buffer_A_;
        size_t buffer_A_pos_;
        char*  buffer_B_;
        size_t buffer_B_pos_;
        size_t size_threshold_;
        size_t time_threshold_; // ms

        mutex mtx_;
        condition_variable cv_push_;
        condition_variable cv_swap_;

    };
}

#endif