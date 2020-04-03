#include "IngPipeline.hh"
#include "IngConfig.hh"
#include "IngImgDatRawQueue.hh"
#include "IngImgDatFetcher.hh"

log4cxx::LoggerPtr diffraflow::IngPipeline::logger_
    = log4cxx::Logger::getLogger("IngPipeline");

diffraflow::IngPipeline::IngPipeline(IngConfig* config) {
    config_obj_ = config;
    running_flag_ = false;
    image_data_raw_queue_ = nullptr;
    image_data_fetcher_ = nullptr;
}

diffraflow::IngPipeline::~IngPipeline() {

}

void diffraflow::IngPipeline::start_run() {
    if (running_flag_) return;

    image_data_raw_queue_ = new IngImgDatRawQueue(config_obj_->imgdat_queue_capacity);
    image_data_fetcher_ = new IngImgDatFetcher(
        config_obj_->combiner_host,
        config_obj_->combiner_port,
        config_obj_->ingester_id,
        image_data_raw_queue_);
    image_data_fetcher_->set_recnxn_policy(
        config_obj_->recnxn_wait_time,
        config_obj_->recnxn_max_count);

    if (image_data_fetcher_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image data fetcher.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image data fetcher.")
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async([this]() {
        image_data_fetcher_->wait();
        image_data_raw_queue_->stop();
        // other steps in the pipeline
    }).wait();

}

void diffraflow::IngPipeline::terminate() {
    if (!running_flag_) return;

    image_data_raw_queue_->stop();

    int result = image_data_fetcher_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image data fetcher is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image data fetcher is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image data fetcher has not yet been started.");
    }

}

