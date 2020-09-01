#include "IngCalibrationWorker.hh"

log4cxx::LoggerPtr diffraflow::IngCalibrationWorker::logger_ = log4cxx::Logger::getLogger("IngCalibrationWorker");

diffraflow::IngCalibrationWorker::IngCalibrationWorker(
    IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out) {
    image_queue_in_ = img_queue_in;
    image_queue_out_ = img_queue_out;
    worker_status_ = kNotStart;
}

diffraflow::IngCalibrationWorker::~IngCalibrationWorker() {}

void diffraflow::IngCalibrationWorker::do_calib_(ImageDataType::Field& image_data) {
    for (size_t i = 0; i < MOD_CNT; i++) {
        if (image_data.alignment[i]) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    switch (image_data.gain_level[i][h][w]) {
                    case 0:
                        image_data.pixel_data[i][h][w] = (image_data.pixel_data[i][h][w] - 0.1) * 1.1;
                        break;
                    case 1:
                        image_data.pixel_data[i][h][w] = (image_data.pixel_data[i][h][w] - 0.1) * 1.1;
                        break;
                    case 2:
                        image_data.pixel_data[i][h][w] = (image_data.pixel_data[i][h][w] - 0.1) * 1.1;
                        break;
                    }
                }
            }
        }
    }
    image_data.calib_level = 1;
}

int diffraflow::IngCalibrationWorker::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        do_calib_(*image_with_feature->image_data);

        // debug
        // image_with_feature->image_data_calib.print();

        if (image_queue_out_->push(image_with_feature)) {
            LOG4CXX_DEBUG(logger_, "pushed the calibrated data into queue.");
        } else {
            break;
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngCalibrationWorker::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(&IngCalibrationWorker::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
    if (worker_status_ == kRunning) {
        return true;
    } else {
        return false;
    }
}

void diffraflow::IngCalibrationWorker::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngCalibrationWorker::stop() {
    if (worker_status_ == kNotStart) {
        return -1;
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    int result = -2;
    if (worker_.valid()) {
        result = worker_.get();
    }
    return result;
}
