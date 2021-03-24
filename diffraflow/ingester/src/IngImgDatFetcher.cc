#include "IngImgDatFetcher.hh"
#include "IngImgFtrBuffer.hh"
#include "PrimitiveSerializer.hh"
#include "ImageDataType.hh"
#include "ImageDataField.hh"
#include "Decoder.hh"
#include <msgpack.hpp>

// 3 MiB
#define MAX_PAYLOAD_SIZE 3145728

log4cxx::LoggerPtr diffraflow::IngImgDatFetcher::logger_ = log4cxx::Logger::getLogger("IngImgDatFetcher");

diffraflow::IngImgDatFetcher::IngImgDatFetcher(string combiner_host, int combiner_port, uint32_t ingester_id,
    IngImgFtrBuffer* buffer, IngBufferItemQueue* queue, bool use_gpu)
    : GenericClient(combiner_host, combiner_port, ingester_id, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE), use_gpu_(use_gpu) {
    image_feature_buffer_ = buffer;
    item_queue_raw_ = queue;
    recnxn_wait_time_ = 0;
    recnxn_max_count_ = 0;
    max_successive_fail_count_ = 5;
    worker_status_ = kNotStart;
}

diffraflow::IngImgDatFetcher::IngImgDatFetcher(
    string combiner_sock, uint32_t ingester_id, IngImgFtrBuffer* buffer, IngBufferItemQueue* queue, bool use_gpu)
    : GenericClient(combiner_sock, ingester_id, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE), use_gpu_(use_gpu) {
    image_feature_buffer_ = buffer;
    item_queue_raw_ = queue;
    recnxn_wait_time_ = 0;
    recnxn_max_count_ = 0;
    max_successive_fail_count_ = 5;
    worker_status_ = kNotStart;
}

diffraflow::IngImgDatFetcher::~IngImgDatFetcher() { stop(); }

void diffraflow::IngImgDatFetcher::set_recnxn_policy(size_t wait_time, size_t max_count) {
    recnxn_wait_time_ = wait_time;
    recnxn_max_count_ = max_count;
}

bool diffraflow::IngImgDatFetcher::connect_to_combiner_() {
    if (client_sock_fd_ > 0) return true;
    if (worker_status_ == kStopped) {
        LOG4CXX_WARN(logger_, "image data fetcher is stopped, abort connecting.");
        return false;
    }
    bool result = false;
    for (size_t current_count = 0; current_count <= recnxn_max_count_; current_count++) {
        if (connect_to_server()) {
            result = true;
            break;
        } else {
            if (current_count < recnxn_max_count_) {
                LOG4CXX_WARN(logger_, "failed to connect to combiner " << get_server_address() << ", wait for "
                                                                       << recnxn_wait_time_ << "ms and retry ("
                                                                       << current_count + 1 << "/" << recnxn_max_count_
                                                                       << ") ...");
                if (recnxn_wait_time_ > 0) {
                    unique_lock<mutex> ulk(cnxn_mtx_);
                    cnxn_cv_.wait_for(ulk, std::chrono::milliseconds(recnxn_wait_time_),
                        [this]() { return worker_status_ == kStopped; });
                }
                if (worker_status_ == kStopped) {
                    LOG4CXX_WARN(logger_, "image data fetcher is stopped, abort reconnecting.");
                    return false;
                }
            }
        }
    }
    if (result) {
        LOG4CXX_INFO(logger_, "successfully connected to combiner: " << get_server_address());
        return true;
    } else {
        LOG4CXX_WARN(logger_, "failed to connect to combiner " << get_server_address() << " after " << recnxn_max_count_
                                                               << " retry counts.");
        return false;
    }
}

diffraflow::IngImgDatFetcher::ReceiveRes diffraflow::IngImgDatFetcher::receive_one_image(
    shared_ptr<IngBufferItem>& item) {

    uint32_t payload_type = 0;
    shared_ptr<vector<char>> payload_data;
    if (!receive_one_(payload_type, payload_data, MAX_PAYLOAD_SIZE)) {
        return kDisconnected;
    }

    if (payload_type != 0xABCDEEEE) {
        LOG4CXX_WARN(logger_, "got unknown payload type: " << payload_type);
        return kFail;
    }

    // decode
    if (ImageDataType::decode(
            *image_feature_buffer_->image_data_host(item->index), payload_data->data(), payload_data->size())) {
        item->rawdata = payload_data;
        return kSucc;
    } else {
        LOG4CXX_WARN(logger_, "failed decode image data.");
        return kFail;
    }
}

int diffraflow::IngImgDatFetcher::run_() {
    int result = 0;
    while (worker_status_ != kStopped && connect_to_combiner_()) {
        worker_status_ = kRunning;
        cv_status_.notify_all();
        size_t successive_fail_count = 0;
        for (bool running = true; running && worker_status_ == kRunning;) {

            int next_index = image_feature_buffer_->next();
            if (next_index < 0) {
                LOG4CXX_WARN(logger_, "image_feature_buffer_ is stopped, close the connection and stop running.");
                close_connection();
                worker_status_ = kStopped;
                result = 0;
                break;
            }

            shared_ptr<IngBufferItem> item = make_shared<IngBufferItem>(next_index);

            switch (receive_one_image(item)) {
            case kDisconnected:
                if (worker_status_ == kStopped) {
                    result = 0;
                    break;
                }
                LOG4CXX_WARN(logger_,
                    "error found when receiving one image from combiner, close the connection and try to reconnect.")
                close_connection();
                running = false;
                break;
            case kSucc:
                successive_fail_count = 0;

                if (item_queue_raw_->push(item)) {
                    LOG4CXX_DEBUG(logger_, "pushed one item into item_queue_raw_.");
                } else {
                    LOG4CXX_WARN(logger_, "item_queue_raw_ is stopped, close the connection and stop running.");
                    close_connection();
                    worker_status_ = kStopped;
                    running = false;
                    result = 0;
                }

                break;
            case kFail:
                successive_fail_count++;
                LOG4CXX_WARN(logger_, "failed to deserialize image data (" << successive_fail_count << "/"
                                                                           << max_successive_fail_count_ << ").");
                if (successive_fail_count >= max_successive_fail_count_) {
                    LOG4CXX_WARN(logger_, "successively failed for "
                                              << successive_fail_count
                                              << " times, close the connection and stop running.");
                    close_connection();
                    worker_status_ = kStopped;
                    running = false;
                    result = 1;
                }
            }
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngImgDatFetcher::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(std::launch::async, &IngImgDatFetcher::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
    if (worker_status_ == kRunning) {
        return true;
    } else {
        return false;
    }
}

void diffraflow::IngImgDatFetcher::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngImgDatFetcher::stop() {
    if (worker_status_ == kNotStart) {
        return -1;
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    cnxn_cv_.notify_all();
    close_connection();
    int result = -2;
    if (worker_.valid()) {
        result = worker_.get();
    }
    return result;
}
