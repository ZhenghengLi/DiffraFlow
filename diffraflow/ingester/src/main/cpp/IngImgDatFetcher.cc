#include "IngImgDatFetcher.hh"

log4cxx::LoggerPtr diffraflow::IngImgDatFetcher::logger_
    = log4cxx::Logger::getLogger("IngImgDatFetcher");

diffraflow::IngImgDatFetcher::IngImgDatFetcher(
    string combiner_host, int combiner_port, uint32_t ingester_id, IngImgDatRawQueue* raw_queue):
    GenericClient(combiner_host, combiner_port, ingester_id, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE) {
    imgdat_raw_queue_ = raw_queue;
    recnxn_wait_time_ = 0;
    recnxn_max_count_ = 0;
}

diffraflow::IngImgDatFetcher::~IngImgDatFetcher() {

}

void diffraflow::IngImgDatFetcher::set_recnxn_policy(size_t wait_time, size_t max_count) {
    recnxn_wait_time_ = wait_time;
    recnxn_max_count_ = max_count;
}

bool diffraflow::IngImgDatFetcher::connect_to_combiner() {
    if (stopped_) {
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
                    << dest_host_ << ":" << dest_port_
                    << ", wait for " << recnxn_wait_time_ << "ms and retry ("
                    << current_count + 1 << "/" << recnxn_max_count_ << ") ...");
                if (recnxn_wait_time_ > 0) {
                    unique_lock<mutex> ulk(cnxn_mtx_);
                    cnxn_cv_.wait_for(ulk, std::chrono::milliseconds(recnxn_wait_time_),
                        [this]() {return stopped_.load();}
                    );
                }
                if (stopped_.load()) {
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
        LOG4CXX_WARN(logger_, "failed to connect to combiner " << dest_host_ << ":" << dest_port_
            << " after " << recnxn_max_count_ << "retry counts.");
        return false;
    }
}
