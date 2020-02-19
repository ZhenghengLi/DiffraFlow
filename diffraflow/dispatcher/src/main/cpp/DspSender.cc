#include "DspSender.hh"
#include "PrimitiveSerializer.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <string>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <snappy.h>

log4cxx::LoggerPtr diffraflow::DspSender::logger_
    = log4cxx::Logger::getLogger("DspSender");

diffraflow::DspSender::DspSender(string hostname, int port, int id, bool compr_flag):
    GenericClient(hostname, port, id, 0xDDCC1234, 0xDDD22CCC) {
    buffer_size_ = 1024 * 1024 * 4 - 16; // 4 MiB - 16 B
    size_threshold_ = 1024 * 1024;  // 1 MiB
    time_threshold_ = 100; // 0.1 second
    buffer_A_ = new char[buffer_size_];
    buffer_A_limit_ = 0;
    buffer_A_imgct_ = 0;
    buffer_B_ = new char[buffer_size_];
    buffer_B_limit_ = 0;
    buffer_B_imgct_ = 0;
    sending_thread_ = nullptr;
    run_flag_ = false;
    compress_flag_ = compr_flag;
}

diffraflow::DspSender::~DspSender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
}

void diffraflow::DspSender::push(const char* data, const size_t len) {
    if (!run_flag_) return;
    unique_lock<mutex> lk(mtx_);
    // wait when there is no enough space
    cv_push_.wait(lk, [&]() {return !run_flag_ || (len + 4 <= buffer_size_ - buffer_A_limit_);});
    if (!run_flag_) return;
    // image frame size
    gPS.serializeValue<uint32_t>(len, buffer_A_ + buffer_A_limit_, 4);
    // image frame data
    std::copy(data, data + len, buffer_A_ + buffer_A_limit_ + 4);
    buffer_A_limit_ += 4 + len;
    buffer_A_imgct_ += 1;
    // foreward limit and check size threshold
    if (buffer_A_limit_ > size_threshold_) {
        cv_swap_.notify_one();
    }
}

bool diffraflow::DspSender::swap_() {
    // swap buffer_A_ and buffer_B_ for async sending
    unique_lock<mutex> lk(mtx_);
    if (buffer_A_limit_ < size_threshold_) {
        cv_swap_.wait_for(lk, std::chrono::milliseconds(time_threshold_));
    }
    if (buffer_A_limit_ > 0) {
        // do the swap
        char* tmp_buff = buffer_B_;
        buffer_B_ = buffer_A_;
        buffer_B_limit_ = buffer_A_limit_;
        buffer_B_imgct_ = buffer_A_imgct_;
        buffer_A_ = tmp_buff;
        buffer_A_limit_ = 0;
        buffer_A_imgct_ = 0;
        cv_push_.notify_one();
        return true;
    } else {
        return false;
    }
}

void diffraflow::DspSender::send_() {
    // send all data in buffer_B_
    lock_guard<mutex> lk_send(mtx_send_);
    if (buffer_B_limit_ > 0) {
        send_buffer_(buffer_B_, buffer_B_limit_, buffer_B_imgct_);
        buffer_B_limit_ = 0;
        buffer_B_imgct_ = 0;
    }
}

void diffraflow::DspSender::send_remaining() {
    // do this only after stopping and before deleting
    lock_guard<mutex> lk_send(mtx_send_);
    lock_guard<mutex> lk(mtx_);
    // send all data in buffer_A
    if (buffer_A_limit_ > 0) {
        send_buffer_(buffer_A_, buffer_A_limit_, buffer_A_imgct_);
        buffer_A_limit_ = 0;
        buffer_A_imgct_ = 0;
    }
}

void diffraflow::DspSender::send_buffer_(const char* buffer, const size_t limit, const size_t imgct) {
    // try to connect if lose connection
    if (client_sock_fd_ < 0) {
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to combiner.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to combiner, discard data in buffer.");
            return;
        }
    }
    // packet format: packet_head(4) | packet_size(4) | image_seq_head(4) | image_count(4) | image_seq_data

    // - send uncompressed data
    uint32_t payload_type = 0xABCDFF00;
    const char* current_buffer = buffer;
    size_t current_limit = limit;

    // - send compressed data if compress_flag is true
    std::string compressed_str;
    if (compress_flag_) {
        payload_type = 0xABCDFF01;
        snappy::Compress(buffer, limit, &compressed_str);
        current_buffer = compressed_str.data();
        current_limit = compressed_str.size();
    }

    LOG4CXX_DEBUG(logger_, "raw size = " << limit << ", sent size = " << current_limit);

    // send head
    char head_buffer[16];
    gPS.serializeValue<uint32_t>(sending_head_, head_buffer, 4);
    gPS.serializeValue<uint32_t>(8 + current_limit, head_buffer + 4, 4);
    gPS.serializeValue<uint32_t>(payload_type, head_buffer + 8, 4);
    gPS.serializeValue<uint32_t>(imgct, head_buffer + 12, 4);
    for (size_t pos = 0; pos < 16;) {
        int count = write(client_sock_fd_, head_buffer + pos, 16 - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger_, "error found when sending data: " << strerror(errno));
            close_connection();
            return;
        } else {
            pos += count;
        }
    }
    LOG4CXX_INFO(logger_, "done a write for head.");

    // send data in current_buffer
    for (size_t pos = 0; pos < current_limit;) {
        int count = write(client_sock_fd_, current_buffer + pos, current_limit - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger_, "error found when sending data: " << strerror(errno));
            close_connection();
            return;
        } else {
            pos += count;
        }
    }
    LOG4CXX_INFO(logger_, "done a write for data.");
}

void diffraflow::DspSender::start() {
    if (run_flag_) return;
    run_flag_ = true;
    if (sending_thread_ != nullptr) return;
    sending_thread_ = new thread(
        [this]() {
            while (run_flag_) {
                if (swap_()) send_();
            }
        }
    );
}

void diffraflow::DspSender::stop() {
    if (!run_flag_) return;
    run_flag_ = false;
    if (sending_thread_ != nullptr) {
        sending_thread_->join();
        delete sending_thread_;
        sending_thread_ = nullptr;
    }
    cv_push_.notify_one();
}
