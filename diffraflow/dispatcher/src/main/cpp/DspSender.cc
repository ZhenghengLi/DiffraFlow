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
#include <lz4.h>
#include <zstd.h>

log4cxx::LoggerPtr diffraflow::DspSender::logger_
    = log4cxx::Logger::getLogger("DspSender");

diffraflow::DspSender::DspSender(string hostname, int port, int id,
    CompressMethod compr_method, int compr_level):
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
    buffer_compress_ = new char[buffer_size_];
    buffer_compress_limit_ = 0;
    sending_thread_ = nullptr;
    run_flag_ = false;
    compress_method_ = compr_method;
    if (compress_method_ == kZSTD) {
        compress_level_ = ( (compr_level >= 1 && compr_level < 20) ? compr_level : 1);
    }
}

diffraflow::DspSender::~DspSender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
    delete [] buffer_compress_;
}

void diffraflow::DspSender::push(const char* data, const size_t len) {
    if (!run_flag_) return;
    unique_lock<mutex> lk(mtx_swap_);
    // wait when there is no enough space
    cv_push_.wait(lk, [&]() {return !run_flag_ || (len + 4 <= buffer_size_ - buffer_A_limit_);});
    if (!run_flag_) return;
    // image frame size
    gPS.serializeValue<uint32_t>(len, buffer_A_ + buffer_A_limit_, 4);
    // image frame data
    std::copy(data, data + len, buffer_A_ + buffer_A_limit_ + 4);
    // foreward limit and check size threshold
    buffer_A_limit_ += 4 + len;
    buffer_A_imgct_ += 1;
    if (buffer_A_limit_ > size_threshold_) {
        cv_swap_.notify_one();
    }
}

bool diffraflow::DspSender::swap_() {
    // swap buffer_A_ and buffer_B_ for async sending
    unique_lock<mutex> lk(mtx_swap_);
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
    lock_guard<mutex> lk(mtx_swap_);
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
    const char* payload_data_buffer = buffer;
    size_t payload_data_size = limit;

    // - send compressed data if compress_method_ is not kNone
    switch (compress_method_) {
    case kLZ4:
        payload_type = 0xABCDFF01;
        buffer_compress_limit_ = LZ4_compress_default(buffer, buffer_compress_, limit, buffer_size_);
        if (buffer_compress_limit_ == 0) {
            LOG4CXX_WARN(logger_, "failed to compress data by LZ4, discard data in buffer.");
            return;
        }
        payload_data_buffer = buffer_compress_;
        payload_data_size   = buffer_compress_limit_;
        break;
    case kSnappy:
        payload_type = 0xABCDFF02;
        snappy::RawCompress(buffer, limit, buffer_compress_, &buffer_compress_limit_);
        payload_data_buffer = buffer_compress_;
        payload_data_size   = buffer_compress_limit_;
        break;
    case kZSTD:
        payload_type = 0xABCDFF03;
        buffer_compress_limit_ = ZSTD_compress(buffer_compress_, buffer_size_, buffer, limit, compress_level_);
        if (ZSTD_isError(buffer_compress_limit_)) {
            LOG4CXX_WARN(logger_, "failed to compress data by ZSTD with error: "
                << ZSTD_getErrorName(buffer_compress_limit_) << ", discard data in buffer.");
            return;
        }
        payload_data_buffer = buffer_compress_;
        payload_data_size   = buffer_compress_limit_;
        break;
    }

    LOG4CXX_DEBUG(logger_, "raw size = " << limit << ", sent size = " << payload_data_size);

    // send data
    char payload_head_buffer[8];
    gPS.serializeValue<uint32_t>(payload_type, payload_head_buffer, 4);
    gPS.serializeValue<uint32_t>(imgct, payload_head_buffer + 4, 4);
    if (!send_one_(payload_head_buffer, 8, payload_data_buffer, payload_data_size)) {
        close_connection();
    }

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
