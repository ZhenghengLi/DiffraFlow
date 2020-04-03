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

void diffraflow::IngImgDatFetcher::set_recnxn_policy(int wait_time, int max_count) {
    recnxn_wait_time_ = wait_time;
    recnxn_max_count_ = max_count;
}

bool diffraflow::IngImgDatFetcher::connect_to_combiner() {

    return true;
}