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

#include "MetricsProvider.hh"

using std::string;
using std::thread;
using std::atomic;
using std::atomic_int;
using std::mutex;
using std::condition_variable;
using std::vector;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class GenericDgramReceiver : public MetricsProvider {
    public:
        GenericDgramReceiver(string host, int port);
        virtual ~GenericDgramReceiver();

        bool start(int cpu_id = -1);
        int wait();
        int stop_and_close();

    public:
        struct {
            atomic<uint64_t> total_recv_count;
            atomic<uint64_t> total_recv_size;
            atomic<uint64_t> total_error_count;
            atomic<uint64_t> total_processed_count;
        } dgram_metrics;

        virtual json::value collect_metrics() override;

    private:
        void run_();
        thread* worker_thread_;
        atomic_int worker_result_;

    protected:
        enum ReceiverStatus { kNotStart, kRunning, kStopping, kStopped, kClosed };

    protected:
        bool create_udp_sock_();

        virtual void process_datagram_(shared_ptr<vector<char>>& datagram);

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