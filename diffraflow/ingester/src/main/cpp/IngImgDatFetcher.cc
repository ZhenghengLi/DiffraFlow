#include "IngImgDatFetcher.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include <msgpack.hpp>

log4cxx::LoggerPtr diffraflow::IngImgDatFetcher::logger_ = log4cxx::Logger::getLogger("IngImgDatFetcher");

diffraflow::IngImgDatFetcher::IngImgDatFetcher(
    string combiner_host, int combiner_port, uint32_t ingester_id, IngImgWthFtrQueue* queue)
    : GenericClient(combiner_host, combiner_port, ingester_id, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE) {
    imgWthFtrQue_raw_ = queue;
    recnxn_wait_time_ = 0;
    recnxn_max_count_ = 0;
    max_successive_fail_count_ = 5;
    worker_status_ = kNotStart;
    imgdat_buffer_size_ = 16 * 1024 * 1024; // 16MiB
    imgdat_buffer_ = new char[imgdat_buffer_size_];
}

diffraflow::IngImgDatFetcher::~IngImgDatFetcher() {
    stop();
    delete[] imgdat_buffer_;
}

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
                LOG4CXX_WARN(logger_, "failed to connect to combiner "
                                          << dest_host_ << ":" << dest_port_ << ", wait for " << recnxn_wait_time_
                                          << "ms and retry (" << current_count + 1 << "/" << recnxn_max_count_
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
        LOG4CXX_INFO(logger_, "successfully connected to combiner " << dest_host_ << ":" << dest_port_);
        return true;
    } else {
        LOG4CXX_WARN(logger_, "failed to connect to combiner " << dest_host_ << ":" << dest_port_ << " after "
                                                               << recnxn_max_count_ << " retry counts.");
        return false;
    }
}

diffraflow::IngImgDatFetcher::ReceiveRes diffraflow::IngImgDatFetcher::receive_one_image(ImageData& image_data) {
    size_t payload_size = 0;
    if (receive_one_(imgdat_buffer_, imgdat_buffer_size_, payload_size)) {
        LOG4CXX_DEBUG(logger_, "successfully received one image.");
    } else {
        LOG4CXX_WARN(logger_, "failed to receive one image.");
        return kDisconnected;
    }
    // check
    if (payload_size <= 4) {
        LOG4CXX_WARN(logger_, "got too short image data packet.");
        return kFail;
    }
    uint32_t payload_head = gDC.decode_byte<uint32_t>(imgdat_buffer_, 0, 3);
    if (payload_head != 0xABCDEEEE) {
        LOG4CXX_WARN(logger_, "got unknown payload head: " << payload_head);
        return kFail;
    }
    // decode
    if (image_data.decode(imgdat_buffer_ + 4, payload_size - 4)) {
        image_data.print();
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
        size_t successive_fail_count_ = 0;
        for (bool running = true; running && worker_status_ == kRunning;) {
            shared_ptr<ImageWithFeature> image_with_feature = make_shared<ImageWithFeature>();
            switch (receive_one_image(image_with_feature->image_data_raw)) {
            case kDisconnected:
                LOG4CXX_WARN(logger_, "error found when receiving one image from combiner,"
                                          << " close the connection and try to reconnect.")
                close_connection();
                running = false;
                break;
            case kSucc:
                successive_fail_count_ = 0;

                // for debug
                // image_with_feature->image_data_raw.print();

                // if (imgWthFtrQue_raw_->push(image_with_feature)) {
                //     LOG4CXX_DEBUG(logger_, "pushed one image into imgdat_raw_queue_.");
                // } else {
                //     LOG4CXX_WARN(logger_, "raw image data queue is stopped,"
                //                               << " close the connection and stop running.");
                //     close_connection();
                //     worker_status_ = kStopped;
                //     running = false;
                // }

                break;
            case kFail:
                successive_fail_count_++;
                LOG4CXX_WARN(logger_, "failed to deserialize image data (" << successive_fail_count_ << "/"
                                                                           << max_successive_fail_count_ << ").");
                if (successive_fail_count_ >= max_successive_fail_count_) {
                    LOG4CXX_WARN(logger_, "successively failed for "
                                              << successive_fail_count_
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
    worker_ = async(&IngImgDatFetcher::run_, this);
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
    close_connection();
    int result = -2;
    if (worker_.valid()) {
        result = worker_.get();
    }
    return result;
}
