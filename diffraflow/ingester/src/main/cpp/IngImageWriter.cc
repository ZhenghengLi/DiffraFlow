#include "IngImageWriter.hh"
#include "IngImgWthFtrQueue.hh"
#include "IngConfig.hh"

log4cxx::LoggerPtr diffraflow::IngImageWriter::logger_
    = log4cxx::Logger::getLogger("IngImageWriter");

diffraflow::IngImageWriter::IngImageWriter(
    IngImgWthFtrQueue* img_queue_in, IngConfig* conf_obj) {
    image_queue_in_ = img_queue_in;
    config_obj_ = conf_obj;
    worker_status_ = kNotStart;

    current_run_number_ = config_obj_->get_dy_run_number();
    current_turn_number_ = 0;
    current_sequence_number_ = 0;

    image_file_hdf5_ = new ImageFileHDF5W(config_obj_->hdf5_buffer_size, config_obj_->hdf5_chunk_size, true);
    image_file_raw_ = new ImageFileRawW();

}

diffraflow::IngImageWriter::~IngImageWriter() {
    delete image_file_hdf5_;
    image_file_hdf5_ = nullptr;
    delete image_file_raw_;
    image_file_raw_ = nullptr;
}

int diffraflow::IngImageWriter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        if (save_image_(image_with_feature)) {
            LOG4CXX_DEBUG(logger_, "saved one image into file.");
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngImageWriter::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(&IngImageWriter::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk,
        [this]() {
            return worker_status_ != kNotStart;
        }
    );
    if (worker_status_ == kRunning) {
        return true;
    } else {
        return false;
    }

}

void diffraflow::IngImageWriter::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngImageWriter::stop() {
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

bool diffraflow::IngImageWriter::save_image_(const shared_ptr<ImageWithFeature>& image_with_feature) {
    return true;
}

bool diffraflow::IngImageWriter::create_directories_() {

    return true;
}

bool diffraflow::IngImageWriter::open_hdf5_file_() {

    return true;
}

bool diffraflow::IngImageWriter::open_raw_file_() {

    return true;
}

bool diffraflow::IngImageWriter::open_files_() {

    return true;
}

void diffraflow::IngImageWriter::close_files_() {

}
