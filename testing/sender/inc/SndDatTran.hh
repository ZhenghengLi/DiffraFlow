#ifndef __SndDatTran_H__
#define __SndDatTran_H__

#include <log4cxx/logger.h>
#include <fstream>
#include <mutex>
#include <atomic>
#include <thread>

#include "MetricsProvider.hh"
#include "BlockingQueue.hh"

using std::ifstream;
using std::string;
using std::mutex;
using std::atomic;
using std::atomic_bool;
using std::thread;

namespace diffraflow {

    class SndConfig;
    class SndTcpSender;
    class SndUdpSender;

    class SndDatTran : public MetricsProvider {
    public:
        explicit SndDatTran(SndConfig* conf_obj);
        ~SndDatTran();

        bool create_tcp_sender(string dispatcher_host, int dispatcher_port, uint32_t sender_id, int sender_port = -1);
        bool create_udp_sender(
            string dispatcher_host, int dispatcher_port, int sndbufsize = 4 * 1024 * 1024, int sender_port = -1);
        void delete_sender();

        bool read_and_send(uint32_t event_index);

        bool push_event(uint32_t);
        bool start_sender(int cpu_id = -1);
        void stop_sender();

    public:
        enum SenderType { kTCP, kUDP, kNotSet };

    public:
        struct {
            atomic<uint64_t> invoke_counts;
            atomic<uint64_t> busy_counts;
            atomic<uint64_t> large_index_counts;
            atomic<uint64_t> read_succ_counts;
            atomic<uint64_t> key_match_counts;
            atomic<uint64_t> send_succ_counts;
            atomic<uint64_t> send_fail_counts;
            atomic<uint64_t> read_send_succ_counts;
            atomic<uint64_t> read_send_fail_counts;
        } transfer_metrics;

        json::value collect_metrics() override;

    private:
        SenderType sender_type_;
        SndTcpSender* tcp_sender_;
        SndUdpSender* udp_sender_;

        BlockingQueue<uint32_t> event_queue_;
        thread* sender_thread_;

        SndConfig* config_obj_;
        char* frame_buffer_;
        char* string_buffer_;

        int current_file_index_;
        ifstream* current_file_;
        string current_file_path_;
        mutex data_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif