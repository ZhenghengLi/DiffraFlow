#include "ImageFileRawW.hh"
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <PrimitiveSerializer.hh>

namespace bf = boost::filesystem;
namespace bs = boost::system;

using std::string;
using std::endl;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::ImageFileRawW::logger_ = log4cxx::Logger::getLogger("ImageFileRawW");

const string diffraflow::ImageFileRawW::inprogress_suffix_ = ".inprogress";

diffraflow::ImageFileRawW::ImageFileRawW() { image_counts_ = 0; }

diffraflow::ImageFileRawW::~ImageFileRawW() { close(); }

bool diffraflow::ImageFileRawW::open(const char* filename) {
    lock_guard<mutex> lg(file_op_mtx_);

    if (outfile_.is_open()) {
        LOG4CXX_ERROR(logger_, "raw data file is alread opened.");
        return false;
    }

    outfile_.open(string(filename) + inprogress_suffix_);
    if (outfile_.is_open()) {
        current_filename_ = filename;
        image_counts_ = 0;
        return true;
    } else {
        return false;
    }
}

void diffraflow::ImageFileRawW::close() {
    lock_guard<mutex> lg(file_op_mtx_);

    if (!outfile_.is_open()) {
        LOG4CXX_WARN(logger_, "raw data file is not opened, and it is needed to close.");
        return;
    }

    outfile_.flush();
    outfile_.close();
    bf::path file_current_path(current_filename_ + inprogress_suffix_);
    bf::path file_target_path(current_filename_);
    bs::error_code ec;
    if (bf::exists(file_current_path)) {
        bf::rename(file_current_path, file_target_path, ec);
        if (ec != bs::errc::success) {
            LOG4CXX_WARN(
                logger_, "failed to rename file " << file_current_path.string() << " with error: " << ec.message());
        }
    }
}

bool diffraflow::ImageFileRawW::write(const char* data, size_t len) {
    if (len > 104857600 /* 100 MiB */) {
        LOG4CXX_ERROR(logger_, "the data block to write is too large.");
        return false;
    }

    lock_guard<mutex> lg(file_op_mtx_);

    if (!outfile_.is_open()) {
        LOG4CXX_ERROR(logger_, "raw data file is not opened.");
        return false;
    }

    char head_buffer[8];
    gPS.serializeValue<uint32_t>(0xABCDEEEE, head_buffer, 4);
    gPS.serializeValue<uint32_t>((uint32_t)len, head_buffer + 4, 4);

    outfile_.write(head_buffer, 8);
    if (outfile_.fail()) {
        LOG4CXX_ERROR(logger_, "failed to write raw data head with error: " << strerror(errno));
        return false;
    }
    outfile_.write(data, len);
    if (outfile_.fail()) {
        LOG4CXX_ERROR(logger_, "failed to write raw data block with error: " << strerror(errno));
        return false;
    }

    image_counts_++;
    return true;
}

size_t diffraflow::ImageFileRawW::size() {
    lock_guard<mutex> lg(file_op_mtx_);
    return image_counts_.load();
}
