#include "IngPipeline.hh"
#include "IngConfig.hh"
#include "IngImgWthFtrQueue.hh"
#include "IngImgDatFetcher.hh"
#include "IngCalibrationWorker.hh"
#include "IngFeatureExtracter.hh"
#include "IngImageFilter.hh"

log4cxx::LoggerPtr diffraflow::IngPipeline::logger_
    = log4cxx::Logger::getLogger("IngPipeline");

diffraflow::IngPipeline::IngPipeline(IngConfig* config) {
    config_obj_ = config;
    running_flag_ = false;

    image_data_fetcher_ = nullptr;
    imgWthFtrQue_raw_ = nullptr;

    calibration_worker_ = nullptr;
    imgWthFtrQue_calib_ = nullptr;

    feature_extracter_ = nullptr;
    imgWthFtrQue_feature_ = nullptr;

    image_filter_ = nullptr;
    imgWthFtrQue_write_ = nullptr;

}

diffraflow::IngPipeline::~IngPipeline() {

}

void diffraflow::IngPipeline::start_run() {
    if (running_flag_) return;

    // prepare queues and workers
    bool error_flag = false;

    //// fetch image data
    imgWthFtrQue_raw_ = new IngImgWthFtrQueue(config_obj_->imgdat_queue_capacity);
    image_data_fetcher_ = new IngImgDatFetcher(
        config_obj_->combiner_host,
        config_obj_->combiner_port,
        config_obj_->ingester_id,
        imgWthFtrQue_raw_);
    image_data_fetcher_->set_recnxn_policy(
        config_obj_->recnxn_wait_time,
        config_obj_->recnxn_max_count);

    //// do calibration
    imgWthFtrQue_calib_ = new IngImgWthFtrQueue(config_obj_->imgdat_queue_capacity);
    calibration_worker_ = new IngCalibrationWorker(imgWthFtrQue_raw_, imgWthFtrQue_calib_);

    //// read calibraion parameters files here

    //// do feature extraction
    imgWthFtrQue_feature_ = new IngImgWthFtrQueue(config_obj_->imgdat_queue_capacity);
    feature_extracter_ = new IngFeatureExtracter(imgWthFtrQue_calib_, imgWthFtrQue_feature_);

    //// do data filtering
    imgWthFtrQue_write_ = new IngImgWthFtrQueue(config_obj_->imgdat_queue_capacity);
    image_filter_ = new IngImageFilter(imgWthFtrQue_feature_, imgWthFtrQue_write_, config_obj_);

    //// do data writing

    if (error_flag) {
        LOG4CXX_ERROR(logger_, "error found when preparing queues and workers");
        return;
    }

    // start workers in turn
    if (image_data_fetcher_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image data fetcher.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image data fetcher.")
        return;
    }

    if (calibration_worker_->start()) {
        LOG4CXX_INFO(logger_, "successfully started calibration worker.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start calibration worker.");
        return;
    }

    if (feature_extracter_->start()) {
        LOG4CXX_INFO(logger_, "successfully started feature extracter.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start feature extracter.");
        return;
    }

    if (image_filter_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image filter.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image filter.");
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async([this]() {
        image_data_fetcher_->wait();
        imgWthFtrQue_raw_->stop();

        calibration_worker_->wait();
        imgWthFtrQue_calib_->stop();

        feature_extracter_->wait();
        imgWthFtrQue_feature_->stop();

        image_filter_->wait();
        imgWthFtrQue_write_->stop();

    }).wait();

}

void diffraflow::IngPipeline::terminate() {
    if (!running_flag_) return;

    // stop data fetcher
    int result = image_data_fetcher_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image data fetcher is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image data fetcher is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image data fetcher has not yet been started.");
    }

    // stop data calibration
    imgWthFtrQue_raw_->stop(/* wait_time */);
    result = calibration_worker_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "calibration worker is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "calibration worker is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "calibration worker has not yet been started.");
    }

    // stop feature extraction
    imgWthFtrQue_calib_->stop(/* wait_time */);
    result = feature_extracter_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "feature extracter is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "feature extracter is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "feature extracter has not yet been started.");
    }

    // stop data writer
    imgWthFtrQue_feature_->stop();
    result = image_filter_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image filter is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image filter is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image filter has not yet been started.");
    }

    imgWthFtrQue_write_->stop();

    // delete objects
    delete image_data_fetcher_;
    image_data_fetcher_ = nullptr;

    delete imgWthFtrQue_raw_;
    imgWthFtrQue_raw_ = nullptr;

    delete calibration_worker_;
    calibration_worker_ = nullptr;

    delete imgWthFtrQue_calib_;
    imgWthFtrQue_calib_ = nullptr;

    delete feature_extracter_;
    feature_extracter_ = nullptr;

    delete imgWthFtrQue_feature_;
    imgWthFtrQue_feature_ = nullptr;

    delete image_filter_;
    image_filter_ = nullptr;

    delete imgWthFtrQue_write_;
    imgWthFtrQue_write_ = nullptr;

}
