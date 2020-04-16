#include "ImageFileRawW.hh"
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;
namespace bs = boost::system;

using std::string;
using std::endl;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::ImageFileRawW::logger_
    = log4cxx::Logger::getLogger("ImageFileRawW");

const string diffraflow::ImageFileRawW::inprogress_suffix_ = ".inprogress";

diffraflow::ImageFileRawW::ImageFileRawW() {
    image_counts_ = 0;
}

diffraflow::ImageFileRawW::~ImageFileRawW() {
    close();
}

bool diffraflow::ImageFileRawW::open(const char* filename) {
    lock_guard<mutex> lg(file_op_mtx_);

    if (outfile_.is_open()) {
        LOG4CXX_ERROR(logger_, "raw data file is alread opened.");
        return false;
    }

    outfile_.open(string(filename) + inprogress_suffix_);
    if (outfile_.is_open()) {
        current_filename_ = filename;
        time_t now_time = time(NULL);
        string now_time_string = boost::trim_copy(string(ctime(&now_time)));
        outfile_ << "Image Raw Data File (for test)" << endl;
        outfile_ << "Create Time: " << now_time_string << endl;
        outfile_ << "=============================================" << endl;
        image_counts_ = 0;
        return true;
    } else {
        return false;
    }
}

void diffraflow::ImageFileRawW::close() {
    lock_guard<mutex> lg(file_op_mtx_);
    outfile_.close();
    bf::path file_current_path(current_filename_ + inprogress_suffix_);
    bf::path file_target_path(current_filename_);
    bs::error_code ec;
    if (bf::exists(file_current_path)) {
        bf::rename(file_current_path, file_target_path, ec);
        if (ec != bs::errc::success) {
            LOG4CXX_WARN(logger_, "failed to rename file " << file_current_path.string()
                << " with error: " << ec.message());
        }
    }
}

bool diffraflow::ImageFileRawW::write(const ImageData& image_data) {
    lock_guard<mutex> lg(file_op_mtx_);

    if (!outfile_.is_open()) {
        LOG4CXX_ERROR(logger_, "raw data file is not opened.");
        return false;
    }

    if (image_counts_ > 0) {
        outfile_ << "---------------------------------------------" << endl;
    }

    image_data.print(outfile_);
    image_counts_++;

    return true;
}

size_t diffraflow::ImageFileRawW::size() {
    return image_counts_.load();
}
