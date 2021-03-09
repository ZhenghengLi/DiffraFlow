#include "IngCalibrationWorker.hh"
#include "H5Cpp.h"

log4cxx::LoggerPtr diffraflow::IngCalibrationWorker::logger_ = log4cxx::Logger::getLogger("IngCalibrationWorker");

diffraflow::IngCalibrationWorker::IngCalibrationWorker(
    IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out) {
    image_queue_in_ = img_queue_in;
    image_queue_out_ = img_queue_out;
    worker_status_ = kNotStart;

    // init calibration parameters
    for (size_t m = 0; m < MOD_CNT; m++) {
        for (size_t l = 0; l < LEVEL_CNT; l++) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    calib_gain_[m][l][h][w] = 1.0;
                    calib_pedestal_[m][l][h][w] = 0.0;
                }
            }
        }
    }
}

bool diffraflow::IngCalibrationWorker::read_calib_file(const char* calib_file) {

    H5::Exception::dontPrint();

    try {
        H5::H5File* h5file = new H5::H5File(calib_file, H5F_ACC_RDONLY);

        // gain
        H5::DataSet gain_dset = h5file->openDataSet("gain");
        H5::DataSpace gain_file_space = gain_dset.getSpace();
        hsize_t gain_mem_dim[] = {MOD_CNT, LEVEL_CNT, FRAME_H, FRAME_W};
        hsize_t gain_offset[] = {0, 0, 0, 0};
        gain_file_space.selectHyperslab(H5S_SELECT_SET, gain_mem_dim, gain_offset);
        H5::DataSpace gain_mem_space(4, gain_mem_dim);
        gain_dset.read(calib_gain_, H5::PredType::NATIVE_FLOAT, gain_mem_space, gain_file_space);

        // pedestal
        H5::DataSet pedestal_dset = h5file->openDataSet("pedestal");
        H5::DataSpace pedestal_file_space = pedestal_dset.getSpace();
        hsize_t pedestal_mem_dim[] = {MOD_CNT, LEVEL_CNT, FRAME_H, FRAME_W};
        hsize_t pedestal_offset[] = {0, 0, 0, 0};
        pedestal_file_space.selectHyperslab(H5S_SELECT_SET, pedestal_mem_dim, pedestal_offset);
        H5::DataSpace pedestal_mem_space(4, pedestal_mem_dim);
        pedestal_dset.read(calib_pedestal_, H5::PredType::NATIVE_FLOAT, pedestal_mem_space, pedestal_file_space);

        h5file->close();
        delete h5file;
        h5file = nullptr;

    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when reading calibration parameters from HDF5 file " << calib_file << " : "
                                                                                                 << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when read calibration parameters from HDF5 file: " << calib_file);
        return false;
    }

    // change the unit of gain from ADC/keV to keV/ADC.
    for (size_t m = 0; m < MOD_CNT; m++) {
        for (size_t l = 0; l < LEVEL_CNT; l++) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    if (calib_gain_[m][l][h][w] > 0) {
                        calib_gain_[m][l][h][w] = 1.0 / calib_gain_[m][l][h][w];
                    } else {
                        calib_gain_[m][l][h][w] = 1.0;
                        LOG4CXX_ERROR(logger_, "found invalid gain: calib_gain_[" << m << "][" << l << "][" << h << "]["
                                                                                  << w
                                                                                  << "] = " << calib_gain_[m][l][h][w]);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

diffraflow::IngCalibrationWorker::~IngCalibrationWorker() {}

void diffraflow::IngCalibrationWorker::do_calib_(shared_ptr<ImageWithFeature>& image_with_feature) {

    ImageDataField& image_data = *image_with_feature->image_data;

    for (size_t m = 0; m < MOD_CNT; m++) {
        if (image_data.alignment[m]) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    size_t l = image_data.gain_level[m][h][w];
                    if (l < LEVEL_CNT) {
                        image_data.pixel_data[m][h][w] =
                            (image_data.pixel_data[m][h][w] - calib_pedestal_[m][l][h][w]) * calib_gain_[m][l][h][w];
                    }
                }
            }
        }
    }

    // ==== common noise subtraction test begin ====

    // subtract pedestal
    // for (size_t m = 0; m < MOD_CNT; m++) {
    //     if (image_data.alignment[m]) {
    //         for (size_t h = 0; h < FRAME_H; h++) {
    //             for (size_t w = 0; w < FRAME_W; w++) {
    //                 size_t l = image_data.gain_level[m][h][w];
    //                 if (l < LEVEL_CNT) {
    //                     image_data.pixel_data[m][h][w] = image_data.pixel_data[m][h][w] -
    //                     calib_pedestal_[m][l][h][w];
    //                 }
    //             }
    //         }
    //     }
    // }

    // double common_noise[MOD_CNT];
    // for (size_t m = 0; m < MOD_CNT; m++) {
    //     // calculate common noise
    //     common_noise[m] = 0;
    //     for (size_t h = 0; h < FRAME_H; h++) {
    //         for (size_t w = 0; w < FRAME_W; w++) {
    //             common_noise[m] += image_data.pixel_data[m][h][w];
    //         }
    //     }
    //     common_noise[m] /= 65536.;
    //     // subtract common noise and correct gain
    //     for (size_t h = 0; h < FRAME_H; h++) {
    //         for (size_t w = 0; w < FRAME_W; w++) {
    //             size_t l = image_data.gain_level[m][h][w];
    //             if (l < LEVEL_CNT) {
    //                 double tmp_adc = image_data.pixel_data[m][h][w] - common_noise[m];
    //                 image_data.pixel_data[m][h][w] = image_data.pixel_data[m][h][w] * calib_gain_[m][l][h][w];
    //             }
    //         }
    //     }
    // }

    // ==== common noise subtraction test end ====

    image_data.calib_level = 1;
}

int diffraflow::IngCalibrationWorker::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        do_calib_(image_with_feature);

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
    worker_ = async(std::launch::async, &IngCalibrationWorker::run_, this);
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
