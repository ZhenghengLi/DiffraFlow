#include "IngImageWriter.hh"
#include "IngImgWthFtrQueue.hh"
#include "IngConfig.hh"
#include <regex>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;
namespace bs = boost::system;

#define STR_BUFF_SIZE 256

log4cxx::LoggerPtr diffraflow::IngImageWriter::logger_ = log4cxx::Logger::getLogger("IngImageWriter");

diffraflow::IngImageWriter::IngImageWriter(IngImgWthFtrQueue* img_queue_in, IngConfig* conf_obj) {
    image_queue_in_ = img_queue_in;
    config_obj_ = conf_obj;
    worker_status_ = kNotStart;

    current_run_number_ = config_obj_->get_dy_run_number();
    current_turn_number_ = 0;
    current_sequence_number_ = 0;
    current_saved_counts_ = 0;
    total_saved_counts_ = 0;
    total_opened_counts_ = 0;

    image_file_hdf5_ = nullptr;
    image_file_raw_ = nullptr;
}

diffraflow::IngImageWriter::~IngImageWriter() { close_files_(); }

int diffraflow::IngImageWriter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        if (config_obj_->storage_dir.empty()) {
            continue;
        }
        // if run number is changed, create new folders
        int new_run_number = config_obj_->get_dy_run_number();
        if (new_run_number != current_run_number_.load()) {
            LOG4CXX_INFO(logger_, "run number changed from "
                                      << current_run_number_.load() << " to " << new_run_number
                                      << ". Create new folders (if not exists) for new run number.");
            current_run_number_ = new_run_number;
            close_files_();
            if (!create_directories_()) {
                LOG4CXX_WARN(logger_, "failed to create new folders for new run number.");
            }
            if (!open_files_()) {
                LOG4CXX_WARN(logger_, "failed to open data files after run number changed.");
            }
        }
        if (save_image_(image_with_feature)) {
            if (current_saved_counts_.load() >= config_obj_->file_imgcnt_limit) {
                LOG4CXX_INFO(logger_, "file limit reached, reopen new files.");
                close_files_();
                open_files_();
            }
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
    if (!config_obj_->storage_dir.empty()) {
        // create folders
        if (!create_directories_()) {
            LOG4CXX_ERROR(logger_, "failed to create directories at start.");
            worker_status_ = kStopped;
            return false;
        }
        // open files
        if (!open_files_()) {
            LOG4CXX_ERROR(logger_, "failed to open data files at start.");
            worker_status_ = kStopped;
            return false;
        }
    }

    worker_ = async(&IngImageWriter::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
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
    close_files_();
    return result;
}

bool diffraflow::IngImageWriter::save_image_(const shared_ptr<ImageWithFeature>& image_with_feature) {
    bool failed = false;
    if (image_file_raw_ != nullptr) {
        if (image_file_raw_->write(image_with_feature->image_data_raw)) {
            LOG4CXX_DEBUG(logger_, "saved one image into raw data file.");
        } else {
            LOG4CXX_WARN(logger_, "failed to save one image into raw data file.");
            failed = true;
        }
    }
    if (image_file_hdf5_ != nullptr) {
        if (image_file_hdf5_->append(image_with_feature->image_data_raw)) {
            LOG4CXX_DEBUG(logger_, "saved one image into hdf5 file.");
        } else {
            LOG4CXX_WARN(logger_, "failed to save one image into hdf5 file.");
            failed = true;
        }
    }
    if (failed) {
        return false;
    } else {
        current_saved_counts_++;
        total_saved_counts_++;
        return true;
    }
}

bool diffraflow::IngImageWriter::create_directories_() {
    // check storage_dir
    bf::path folder_path(config_obj_->storage_dir);
    if (!bf::exists(folder_path)) {
        LOG4CXX_ERROR(logger_, "path " << folder_path.c_str() << " does not exist.");
        return false;
    }
    if (!bf::is_directory(folder_path)) {
        LOG4CXX_ERROR(logger_, "path " << folder_path.c_str() << " is not a directory.");
        return false;
    }
    char str_buffer[STR_BUFF_SIZE];
    bs::error_code ec;
    // R0000
    snprintf(str_buffer, STR_BUFF_SIZE, "R%04d", current_run_number_.load());
    folder_path /= str_buffer;
    if (!bf::exists(folder_path)) {
        bf::create_directory(folder_path, ec);
        if (ec == bs::errc::success) {
            LOG4CXX_INFO(logger_, "created folder " << folder_path.c_str());
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to create folder " << folder_path.c_str() << " with error: " << ec.message());
            return false;
        }
    }
    // R0000/NODENAME_N00
    snprintf(str_buffer, STR_BUFF_SIZE, "%s_N%02d", config_obj_->node_name.c_str(), config_obj_->ingester_id);
    folder_path /= str_buffer;
    if (!bf::exists(folder_path)) {
        bf::create_directory(folder_path, ec);
        if (ec == bs::errc::success) {
            LOG4CXX_INFO(logger_, "created folder " << folder_path.c_str());
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to create folder " << folder_path.c_str() << " with error: " << ec.message());
            return false;
        }
    }
    // R0000/NODENAME_N00/T00
    std::regex turn_regex("T(\\d+)");
    int max_turn_number = -1;
    for (bf::directory_entry& de : bf::directory_iterator(folder_path)) {
        if (!bf::is_directory(de.path())) continue;
        string filename = de.path().filename().string();
        std::smatch match_res;
        if (std::regex_match(filename, match_res, turn_regex)) {
            int cur_turn_number = atoi(match_res[1].str().c_str());
            if (cur_turn_number > max_turn_number) {
                max_turn_number = cur_turn_number;
            }
        }
    }
    current_turn_number_ = max_turn_number + 1;
    snprintf(str_buffer, STR_BUFF_SIZE, "T%02d", current_turn_number_.load());
    folder_path /= str_buffer;
    bf::create_directory(folder_path, ec);
    if (ec == bs::errc::success) {
        LOG4CXX_INFO(logger_, "created folder " << folder_path.c_str());
    } else {
        LOG4CXX_ERROR(logger_, "failed to create folder " << folder_path.c_str() << " with error: " << ec.message());
        return false;
    }
    current_folder_path_string_ = folder_path.string();
    // S0000
    current_sequence_number_ = -1;

    return true;
}

bool diffraflow::IngImageWriter::open_hdf5_file_() {
    if (image_file_hdf5_ != nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is already opened.");
        return false;
    }
    char str_buffer[STR_BUFF_SIZE];
    snprintf(str_buffer, STR_BUFF_SIZE, "R%04d_%s_N%02d_T%02d_S%04d.h5", current_run_number_.load(),
        config_obj_->node_name.c_str(), config_obj_->ingester_id, current_turn_number_.load(),
        current_sequence_number_.load());
    bf::path file_path(current_folder_path_string_);
    file_path /= str_buffer;
    if (bf::exists(file_path)) {
        LOG4CXX_ERROR(logger_, "file " << file_path.c_str() << " already exists.");
        return false;
    }
    // open file
    image_file_hdf5_ =
        new ImageFileHDF5W(config_obj_->hdf5_buffer_size, config_obj_->hdf5_chunk_size, config_obj_->hdf5_swmr_mode);
    if (image_file_hdf5_->open(file_path.c_str(), config_obj_->hdf5_compress_level)) {
        LOG4CXX_INFO(logger_, "successfully opened hdf5 file: " << file_path.c_str());
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "failed to create hdf5 file: " << file_path.c_str());
        return false;
    }
}

bool diffraflow::IngImageWriter::open_raw_file_() {
    if (image_file_raw_ != nullptr) {
        LOG4CXX_ERROR(logger_, "raw file is already opened.");
        return false;
    }
    char str_buffer[STR_BUFF_SIZE];
    snprintf(str_buffer, STR_BUFF_SIZE, "R%04d_%s_N%02d_T%02d_S%04d.dat", current_run_number_.load(),
        config_obj_->node_name.c_str(), config_obj_->ingester_id, current_turn_number_.load(),
        current_sequence_number_.load());
    bf::path file_path(current_folder_path_string_);
    file_path /= str_buffer;
    if (bf::exists(file_path)) {
        LOG4CXX_ERROR(logger_, "file " << file_path.c_str() << " already exists.");
        return false;
    }
    // open file
    image_file_raw_ = new ImageFileRawW();
    if (image_file_raw_->open(file_path.c_str())) {
        LOG4CXX_INFO(logger_, "successfully opened raw data file: " << file_path.c_str());
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "failed to create raw data file: " << file_path.c_str());
        return false;
    }

    return true;
}

bool diffraflow::IngImageWriter::open_files_() {
    current_sequence_number_++;

    if (!open_hdf5_file_()) {
        return false;
    }

    // if (!open_raw_file_()) {
    //     return false;
    // }

    current_saved_counts_ = 0;
    total_opened_counts_++;
    return true;
}

void diffraflow::IngImageWriter::close_files_() {
    if (image_file_hdf5_ != nullptr) {
        image_file_hdf5_->close();
        delete image_file_hdf5_;
        image_file_hdf5_ = nullptr;
    }
    if (image_file_raw_ != nullptr) {
        image_file_raw_->close();
        delete image_file_raw_;
        image_file_raw_ = nullptr;
    }
}

json::value diffraflow::IngImageWriter::collect_metrics() {
    json::value root_json;
    root_json["current_run_number"] = json::value::number(current_run_number_.load());
    root_json["current_turn_number"] = json::value::number(current_turn_number_.load());
    root_json["current_sequence_number"] = json::value::number(current_sequence_number_.load());
    root_json["current_saved_counts"] = json::value::number(current_saved_counts_.load());
    root_json["total_saved_counts"] = json::value::number(total_saved_counts_.load());
    root_json["total_opened_counts"] = json::value::number(total_opened_counts_.load());
    return root_json;
}
