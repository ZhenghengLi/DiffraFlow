#include "IngPipeline.hh"
#include "IngConfig.hh"
#include "IngImgFtrBuffer.hh"
#include "IngImgDatFetcher.hh"
#include "IngCalibrationWorker.hh"
#include "IngFeatureExtracter.hh"
#include "IngImageFilter.hh"
#include "IngImageWriter.hh"
#include "IngImgHttpServer.hh"

#include <cuda_runtime.h>
#include "cudatools.hh"

using std::lock_guard;

log4cxx::LoggerPtr diffraflow::IngPipeline::logger_ = log4cxx::Logger::getLogger("IngPipeline");

diffraflow::IngPipeline::IngPipeline(IngConfig* config) {
    config_obj_ = config;
    running_flag_ = false;

    image_feature_buffer_ = nullptr;

    image_data_fetcher_ = nullptr;
    item_queue_raw_ = nullptr;

    calibration_worker_ = nullptr;
    item_queue_calib_ = nullptr;

    feature_extracter_ = nullptr;
    item_queue_feature_ = nullptr;

    image_filter_ = nullptr;
    item_queue_write_ = nullptr;

    image_http_server_ = nullptr;

    image_writer_ = nullptr;
}

diffraflow::IngPipeline::~IngPipeline() {}

void diffraflow::IngPipeline::start_run() {
    if (running_flag_) return;

    // select gpu device
    if (config_obj_->gpu_enable) {
        LOG4CXX_INFO(logger_, "Use GPU for data processing.");
        cudaError_t cuda_err = cudaSetDevice(config_obj_->gpu_device_index);
        if (cuda_err == cudaSuccess) {
            LOG4CXX_INFO(
                logger_, "Successfully selected " << cudatools::get_device_string(config_obj_->gpu_device_index));
        } else {
            LOG4CXX_ERROR(logger_, "Failed to select GPU of device index " << config_obj_->gpu_device_index);
            return;
        }
    } else {
        LOG4CXX_INFO(logger_, "Use CPU for data processing.");
    }

    //======================================================
    // create buffer
    image_feature_buffer_ = new IngImgFtrBuffer(config_obj_->buffer_capacity, config_obj_->gpu_enable);
    if (!image_feature_buffer_->mem_ready()) {
        LOG4CXX_ERROR(logger_, "Failed to create image feature buffer.");
        return;
    }

    // create all queues and workers
    //// image fetcher
    item_queue_raw_ = new IngBufferItemQueue(config_obj_->queue_capacity_raw);
    if (config_obj_->combiner_sock.empty()) {
        image_data_fetcher_ = new IngImgDatFetcher(config_obj_->combiner_host, config_obj_->combiner_port,
            config_obj_->ingester_id, image_feature_buffer_, item_queue_raw_);
    } else {
        image_data_fetcher_ = new IngImgDatFetcher(
            config_obj_->combiner_sock, config_obj_->ingester_id, image_feature_buffer_, item_queue_raw_);
    }

    //// calibration worker
    item_queue_calib_ = new IngBufferItemQueue(config_obj_->queue_capacity_calib);
    calibration_worker_ = new IngCalibrationWorker(image_feature_buffer_, item_queue_raw_, item_queue_calib_,
        config_obj_->gpu_enable, config_obj_->gpu_device_index);

    //// feature extracter
    item_queue_feature_ = new IngBufferItemQueue(config_obj_->queue_capacity_feature);
    feature_extracter_ = new IngFeatureExtracter(image_feature_buffer_, item_queue_calib_, item_queue_feature_,
        config_obj_->gpu_enable, config_obj_->gpu_device_index);

    //// image filter
    item_queue_write_ = new IngBufferItemQueue(config_obj_->queue_capacity_write);
    image_filter_ = new IngImageFilter(image_feature_buffer_, item_queue_feature_, item_queue_write_, config_obj_,
        config_obj_->gpu_enable, config_obj_->gpu_device_index);

    //// image writer
    image_writer_ = new IngImageWriter(image_feature_buffer_, item_queue_write_, config_obj_);

    //// http server
    image_http_server_ = new IngImgHttpServer(image_feature_buffer_, config_obj_->ingester_id);

    //======================================================
    // config and prepare before start
    //// image fetcher
    image_data_fetcher_->set_recnxn_policy(config_obj_->recnxn_wait_time, config_obj_->recnxn_max_count);

    //// calibration worker
    //// read calibraion parameters files here
    if (!config_obj_->calib_param_file.empty()) {
        if (calibration_worker_->read_calib_file(config_obj_->calib_param_file.c_str())) {
            LOG4CXX_INFO(
                logger_, "successfully read calibration parameters from file: " << config_obj_->calib_param_file);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to read calibration parameters from file: " << config_obj_->calib_param_file);
            return;
        }
    }

    //======================================================
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

    if (image_http_server_->start(config_obj_->image_http_host, config_obj_->image_http_port)) {
        LOG4CXX_INFO(logger_, "successfully started HTTP server listening " << config_obj_->image_http_host << ":"
                                                                            << config_obj_->image_http_port);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start HTTP server.");
        return;
    }

    if (image_writer_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image writer.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image writer.");
        return;
    }

    // start metrics reporter
    metrics_reporter_.add("configuration", config_obj_);
    metrics_reporter_.add("image_data_fetcher", image_data_fetcher_);
    metrics_reporter_.add("image_filter", image_filter_);
    metrics_reporter_.add("image_writer", image_writer_);
    metrics_reporter_.add("image_http_server", image_http_server_);
    if (config_obj_->metrics_pulsar_params_are_set()) {
        if (metrics_reporter_.start_msg_producer(config_obj_->metrics_pulsar_broker_address,
                config_obj_->metrics_pulsar_topic_name, config_obj_->metrics_pulsar_message_key,
                config_obj_->metrics_pulsar_report_period)) {
            LOG4CXX_INFO(logger_, "Successfully started pulsar producer to periodically report metrics.");
        } else {
            LOG4CXX_ERROR(logger_, "Failed to start pulsar producer to periodically report metrics.");
            return;
        }
    }
    if (config_obj_->metrics_http_params_are_set()) {
        if (metrics_reporter_.start_http_server(config_obj_->metrics_http_host, config_obj_->metrics_http_port)) {
            LOG4CXX_INFO(logger_, "Successfully started http server for metrics service.");
        } else {
            LOG4CXX_ERROR(logger_, "Failed to start http server for metrics service.");
            return;
        }
    }

    running_flag_ = true;

    //======================================================
    // then wait for finishing
    async(std::launch::async, [this]() {
        lock_guard<mutex> lg(delete_mtx_);

        image_data_fetcher_->wait();
        item_queue_raw_->stop();

        calibration_worker_->wait();
        item_queue_calib_->stop();

        feature_extracter_->wait();
        item_queue_feature_->stop();

        image_filter_->wait();
        item_queue_write_->stop();

        image_writer_->wait();
    }).wait();
}

void diffraflow::IngPipeline::terminate() {
    if (!running_flag_) return;

    // stop metrics reporter
    metrics_reporter_.stop_http_server();
    metrics_reporter_.stop_msg_producer();
    metrics_reporter_.clear();

    // stop buffer
    image_feature_buffer_->stop();

    // stop data fetcher
    item_queue_raw_->stop();
    int result = image_data_fetcher_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image data fetcher is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image data fetcher is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image data fetcher has not yet been started.");
    }

    // stop data calibration
    item_queue_calib_->stop();
    result = calibration_worker_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "calibration worker is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "calibration worker is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "calibration worker has not yet been started.");
    }

    // stop feature extraction
    item_queue_feature_->stop();
    result = feature_extracter_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "feature extracter is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "feature extracter is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "feature extracter has not yet been started.");
    }

    // stop image filter
    item_queue_write_->stop();
    result = image_filter_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image filter is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image filter is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image filter has not yet been started.");
    }

    // stop image writer
    result = image_writer_->stop();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image writer is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image writer is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image writer has not yet been started.");
    }

    // stop http server
    image_http_server_->stop();

    lock_guard<mutex> lg(delete_mtx_);

    // delete all workers
    delete image_data_fetcher_;
    image_data_fetcher_ = nullptr;

    delete calibration_worker_;
    calibration_worker_ = nullptr;

    delete feature_extracter_;
    feature_extracter_ = nullptr;

    delete image_filter_;
    image_filter_ = nullptr;

    delete image_writer_;
    image_writer_ = nullptr;

    delete image_http_server_;
    image_http_server_ = nullptr;

    // delete all queues
    delete item_queue_raw_;
    item_queue_raw_ = nullptr;

    delete item_queue_calib_;
    item_queue_calib_ = nullptr;

    delete item_queue_feature_;
    item_queue_feature_ = nullptr;

    delete item_queue_write_;
    item_queue_write_ = nullptr;

    delete image_feature_buffer_;
    image_feature_buffer_ = nullptr;

    running_flag_ = false;
}
