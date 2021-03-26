#ifndef GenericConnection_H
#define GenericConnection_H

#include <iostream>
#include <memory>
#include <atomic>
#include <log4cxx/logger.h>

#include "MetricsProvider.hh"
#include "ByteBuffer.hh"

using std::atomic;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class GenericConnection : public MetricsProvider {
    public:
        GenericConnection(int sock_fd, uint32_t greet_hd, uint32_t recv_hd, uint32_t send_hd, size_t buff_sz);
        virtual ~GenericConnection();

        void run(bool receiving_dominant = true);
        bool done();
        void stop();

    public:
        struct {
            atomic<uint64_t> total_sent_size;
            atomic<uint64_t> total_sent_counts;
            atomic<uint64_t> total_received_size;
            atomic<uint64_t> total_received_counts;
            atomic<uint64_t> total_processed_counts;
            atomic<uint64_t> total_skipped_counts;
        } network_metrics;

        virtual json::value collect_metrics() override;

    protected:
        uint32_t connection_id_;
        uint32_t greeting_head_;
        uint32_t receiving_head_;
        uint32_t sending_head_;

        char* buffer_;
        size_t buffer_size_;
        size_t buffer_limit_;
        int client_sock_fd_;

        atomic<bool> done_flag_;

    protected:
        enum ProcessRes { kProcessed, kSkipped, kFailed };

    protected:
        bool send_one_(const char* payload_head_buffer, const size_t payload_head_size, const char* payload_data_buffer,
            const size_t payload_data_size);
        bool send_head_(const uint32_t packet_size);
        bool send_segment_(const char* segment_data_buffer, const size_t segment_data_size);

        bool receive_one_(char* buffer, const size_t buffer_size, size_t& payload_size);
        bool receive_one_(
            uint32_t& payload_type, shared_ptr<ByteBuffer>& payload_data, const uint32_t max_payload_size = 1048576);

        // methods to be implemented
        virtual ProcessRes process_payload_(const char* payload_buffer, const size_t payload_size);
        virtual void after_connected_();
        virtual bool do_preparing_and_sending_();
        virtual bool do_receiving_and_processing_();

    private:
        bool start_connection_();

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
