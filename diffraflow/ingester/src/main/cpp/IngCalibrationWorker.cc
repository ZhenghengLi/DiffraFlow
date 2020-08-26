#include "IngCalibrationWorker.hh"
#include "IngImgWthFtrQueue.hh"

log4cxx::LoggerPtr diffraflow::IngCalibrationWorker::logger_ = log4cxx::Logger::getLogger("IngCalibrationWorker");

diffraflow::IngCalibrationWorker::IngCalibrationWorker(
    IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out) {
    image_queue_in_ = img_queue_in;
    image_queue_out_ = img_queue_out;
    worker_status_ = kNotStart;
}

diffraflow::IngCalibrationWorker::~IngCalibrationWorker() {}

void diffraflow::IngCalibrationWorker::do_calib_(const ImageData& imgdat_raw, ImageData& imgdat_calib) {

    imgdat_calib.bunch_id = imgdat_raw.bunch_id;
    imgdat_calib.late_arrived = imgdat_raw.late_arrived;
    imgdat_calib.alignment_vec = imgdat_raw.alignment_vec;
    imgdat_calib.image_frame_vec.resize(imgdat_raw.image_frame_vec.size());
    for (size_t i = 0; i < imgdat_raw.alignment_vec.size(); i++) {
        if (imgdat_raw.alignment_vec[i]) {

            imgdat_calib.image_frame_vec[i] = make_shared<ImageFrame>();

            // copy meta-data
            imgdat_calib.image_frame_vec[i]->bunch_id = imgdat_raw.image_frame_vec[i]->bunch_id;
            imgdat_calib.image_frame_vec[i]->module_id = imgdat_raw.image_frame_vec[i]->module_id;
            imgdat_calib.image_frame_vec[i]->cell_id = imgdat_raw.image_frame_vec[i]->cell_id;
            imgdat_calib.image_frame_vec[i]->status = imgdat_raw.image_frame_vec[i]->status;

            // copy gain level
            imgdat_calib.image_frame_vec[i]->gain_level.resize(imgdat_raw.image_frame_vec[i]->gain_level.size());
            for (size_t j = 0; j < imgdat_raw.image_frame_vec[i]->gain_level.size(); j++) {
                imgdat_calib.image_frame_vec[i]->gain_level[j] = imgdat_raw.image_frame_vec[i]->gain_level[j];
            }

            // calibrate pixel data, currently just copy
            imgdat_calib.image_frame_vec[i]->pixel_data.resize(imgdat_raw.image_frame_vec[i]->pixel_data.size());
            for (size_t j = 0; j < imgdat_raw.image_frame_vec[i]->pixel_data.size(); j++) {
                switch (imgdat_raw.image_frame_vec[i]->gain_level[j]) {
                case 0:
                    imgdat_calib.image_frame_vec[i]->pixel_data[j] = imgdat_raw.image_frame_vec[i]->pixel_data[j];
                    break;
                case 1:
                    imgdat_calib.image_frame_vec[i]->pixel_data[j] = imgdat_raw.image_frame_vec[i]->pixel_data[j];
                    break;
                case 2:
                    imgdat_calib.image_frame_vec[i]->pixel_data[j] = imgdat_raw.image_frame_vec[i]->pixel_data[j];
                    break;
                default:
                    imgdat_calib.image_frame_vec[i]->pixel_data[j] = imgdat_raw.image_frame_vec[i]->pixel_data[j];
                }
            }
        }
    }
    imgdat_calib.set_calib_level(1);
    imgdat_calib.set_defined();
}

int diffraflow::IngCalibrationWorker::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        do_calib_(image_with_feature->image_data_raw, image_with_feature->image_data_calib);

        // debug
        // image_with_feature->image_data_calib.print();

        // if (image_queue_out_->push(image_with_feature)) {
        //     LOG4CXX_DEBUG(logger_, "pushed the calibrated data into queue.");
        // } else {
        //     break;
        // }
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
