#ifndef __GenericDgramReceiver_H__
#define __GenericDgramReceiver_H__

#include <string>

#include <arpa/inet.h>
#include <log4cxx/logger.h>
#include <future>
#include <atomic>
#include <thread>
#include <vector>
#include <memory>

using std::string;
using std::future;
using std::shared_future;
using std::async;
using std::atomic;
using std::mutex;
using std::condition_variable;
using std::vector;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class GenericDgramReceiver {
    public:
        GenericDgramReceiver(string host, int port);
        ~GenericDgramReceiver();

        bool start();
        void wait();
        int stop_and_close();

    private:
        int run_();
        shared_future<int> worker_;

    protected:
        enum ReceiverStatus { kNotStart, kRunning, kStopped, kClosed };

    protected:
        bool create_udp_sock_();

        virtual void process_datagram(shared_ptr<vector<char>>& datagram);

    protected:
        string receiver_sock_host_;
        int receiver_sock_port_;
        int receiver_sock_fd_;

        struct sockaddr_in receiver_addr_;
        struct sockaddr_in sender_addr_;
        socklen_t sender_addr_len_;

        atomic<ReceiverStatus> receiver_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif