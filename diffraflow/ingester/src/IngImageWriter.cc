#include "IngImageWriter.hh"
#include "IngConfig.hh"
#include "IngImgFtrBuffer.hh"
#include <regex>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <boost/filesystem.hpp>

using std::chrono::duration;
using std::chrono::system_clock;
using std::numeric_limits;

namespace bf = boost::filesystem;
namespace bs = boost::system;

#define STR_BUFF_SIZE 256

log4cxx::LoggerPtr diffraflow::IngImageWriter::logger_ = log4cxx::Logger::getLogger("IngImageWriter");

diffraflow::IngImageWriter::IngImageWriter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngConfig* conf_obj)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), config_obj_(conf_obj) {

    worker_status_ = kNotStart;

    current_run_number_ = config_obj_->get_dy_run_number();
    current_turn_number_ = 0;
    current_sequence_number_ = 0;
    current_imgcnt_limit_ = 0;
    current_saved_counts_ = 0;
    total_saved_counts_ = 0;
    total_opened_counts_ = 0;

    image_file_hdf5_ = nullptr;
    image_file_raw_ = nullptr;
}

diffraflow::IngImageWriter::~IngImageWriter() { close_file_(); }

int diffraflow::IngImageWriter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    IngBufferItem item;

    while (worker_status_ != kStopped && item_queue_in_->take(item)) {

        // debug
        // image_with_feature->image_data_calib.print();

        if (config_obj_->storage_dir.empty()) {
            continue;
        }
        if (!config_obj_->save_calib_data && !config_obj_->save_raw_data) {
            continue;
        }

        // if run number is changed, create new folders
        int new_run_number = config_obj_->get_dy_run_number();
        if (new_run_number != current_run_number_.load()) {
            LOG4CXX_INFO(logger_, "run number changed from "
                                      << current_run_number_.load() << " to " << new_run_number
                                      << ". Create new folders (if not exists) for new run number.");
            current_run_number_ = new_run_number;
            close_file_();
            if (!create_directories_()) {
                LOG4CXX_WARN(logger_, "failed to create new folders for new run number.");
            }
            if (!open_file_()) {
                LOG4CXX_WARN(logger_, "failed to open data files after run number changed.");
            }
        }

        if (save_image_(item)) {
            if (current_saved_counts_.load() >= current_imgcnt_limit_.load()) {
                LOG4CXX_INFO(logger_, "file limit reached, reopen new files.");
                close_file_();
                open_file_();
            }
        }

        image_feature_buffer_->done(item.index);
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
    if (!config_obj_->storage_dir.empty() && (config_obj_->save_calib_data || config_obj_->save_raw_data)) {
        // create folders
        if (!create_directories_()) {
            LOG4CXX_ERROR(logger_, "failed to create directories at start.");
            worker_status_ = kStopped;
            return false;
        }
        // open files
        if (!open_file_()) {
            LOG4CXX_ERROR(logger_, "failed to open data files at start.");
            worker_status_ = kStopped;
            return false;
        }
    }

    worker_ = async(std::launch::async, &IngImageWriter::run_, this);
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
    close_file_();
    return result;
}

bool diffraflow::IngImageWriter::save_image_(const IngBufferItem& item) {
    if (!config_obj_->save_calib_data && !config_obj_->save_raw_data) {
        return false;
    }

    if (config_obj_->save_calib_data && image_file_hdf5_ == nullptr) {
        return false;
    }
    if (config_obj_->save_raw_data && image_file_raw_ == nullptr) {
        return false;
    }

    if (config_obj_->save_calib_data) {
        if (image_file_hdf5_->write(*image_feature_buffer_->image_data_host(item.index))) {
            LOG4CXX_DEBUG(logger_, "saved one image into hdf5 file.");
        } else {
            LOG4CXX_WARN(logger_, "failed to save one image into hdf5 file.");
            return false;
        }
    }
    if (config_obj_->save_raw_data) {
        if (image_file_raw_->write(item.rawdata->data(), item.rawdata->size())) {
            LOG4CXX_DEBUG(logger_, "saved one image into raw data file.");
        } else {
            LOG4CXX_WARN(logger_, "failed to save one image into raw data file.");
            return false;
        }
    }

    current_saved_counts_++;
    total_saved_counts_++;
    return true;
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

bool diffraflow::IngImageWriter::open_file_() {
    if (!config_obj_->save_calib_data && !config_obj_->save_raw_data) {
        return false;
    }

    if (config_obj_->save_calib_data && image_file_hdf5_ != nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is already opened.");
        return false;
    }
    if (config_obj_->save_raw_data && image_file_raw_ != nullptr) {
        LOG4CXX_ERROR(logger_, "raw file is already opened.");
        return false;
    }

    current_sequence_number_++;

    current_imgcnt_limit_ = config_obj_->file_imgcnt_limit;

    if (config_obj_->file_imgcnt_rand > 1) {
        duration<double, std::nano> current_time = system_clock::now().time_since_epoch();
        uint32_t seed = (uint64_t)current_time.count() % numeric_limits<uint32_t>::max();
        srand(seed);
        current_imgcnt_limit_ += rand() % config_obj_->file_imgcnt_rand;
    }

    // construct file path
    char str_buffer[STR_BUFF_SIZE];
    snprintf(str_buffer, STR_BUFF_SIZE, "R%04d_%s_N%02d_T%02d_S%04d", current_run_number_.load(),
        config_obj_->node_name.c_str(), config_obj_->ingester_id, current_turn_number_.load(),
        current_sequence_number_.load());
    bf::path file_path_hdf5(current_folder_path_string_);
    file_path_hdf5 /= string(str_buffer) + ".h5";
    bf::path file_path_raw(current_folder_path_string_);
    file_path_raw /= string(str_buffer) + ".dat";

    // check existance
    if (bf::exists(file_path_hdf5)) {
        LOG4CXX_ERROR(logger_, "file " << file_path_hdf5.c_str() << " already exists.");
        return false;
    }
    if (bf::exists(file_path_raw)) {
        LOG4CXX_ERROR(logger_, "file " << file_path_raw.c_str() << " already exists.");
        return false;
    }

    // open file
    if (config_obj_->save_calib_data) {
        image_file_hdf5_ = new ImageFileHDF5W(config_obj_->hdf5_chunk_size, config_obj_->hdf5_swmr_mode);
        if (image_file_hdf5_->open(file_path_hdf5.c_str(), config_obj_->hdf5_compress_level)) {
            LOG4CXX_INFO(logger_, "successfully opened hdf5 file: " << file_path_hdf5.c_str());
        } else {
            LOG4CXX_ERROR(logger_, "failed to create hdf5 file: " << file_path_hdf5.c_str());
            image_file_hdf5_->close();
            delete image_file_hdf5_;
            image_file_hdf5_ = nullptr;
            return false;
        }
    }
    if (config_obj_->save_raw_data) {
        image_file_raw_ = new ImageFileRawW();
        if (image_file_raw_->open(file_path_raw.c_str())) {
            LOG4CXX_INFO(logger_, "successfully opened raw file: " << file_path_raw.c_str());
        } else {
            LOG4CXX_ERROR(logger_, "failed to create raw file: " << file_path_raw.c_str());
            image_file_raw_->close();
            delete image_file_raw_;
            image_file_raw_ = nullptr;
            return false;
        }
    }

    current_saved_counts_ = 0;
    total_opened_counts_++;

    return true;
}

void diffraflow::IngImageWriter::close_file_() {
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
