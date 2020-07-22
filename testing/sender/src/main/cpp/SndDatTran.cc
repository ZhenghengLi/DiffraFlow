#include "SndDatTran.hh"
#include "SndConfig.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"

#include <cstdio>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/filesystem.hpp>

#define HEAD_SIZE 4
#define FRAME_SIZE 131096
#define STRING_LEN 512

namespace bf = boost::filesystem;
namespace bs = boost::system;

log4cxx::LoggerPtr diffraflow::SndDatTran::logger_ = log4cxx::Logger::getLogger("SndDatTran");

diffraflow::SndDatTran::SndDatTran(SndConfig* conf_obj)
    : GenericClient(conf_obj->dispatcher_host, conf_obj->dispatcher_port, conf_obj->sender_id, 0xFFDD1234, 0xFFF22DDD,
          0xDDD22FFF) {
    config_obj_ = conf_obj;
    head_buffer_ = new char[HEAD_SIZE];
    gPS.serializeValue<uint32_t>(0xABCDFFFF, head_buffer_, HEAD_SIZE);
    frame_buffer_ = new char[FRAME_SIZE];
    string_buffer_ = new char[STRING_LEN];
    current_file_ = nullptr;
    current_file_index_ = -1;
}

diffraflow::SndDatTran::~SndDatTran() {
    delete[] head_buffer_;
    delete[] frame_buffer_;
    delete[] string_buffer_;
    if (current_file_ != nullptr) {
        current_file_->close();
        delete current_file_;
        current_file_ = nullptr;
    }
}

bool diffraflow::SndDatTran::read_and_send(uint32_t event_index) {

    if (event_index >= config_obj_->total_events) {
        return false;
    }

    // try to connect if lose connection
    if (client_sock_fd_ < 0) {
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to dispatcher.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to dispatcher.");
            return false;
        }
    }

    int file_index = event_index / config_obj_->events_per_file;
    int64_t file_offset = (event_index % config_obj_->events_per_file);
    file_offset *= FRAME_SIZE;
    if (file_index != current_file_index_) {
        if (current_file_ != nullptr) {
            current_file_->close();
            delete current_file_;
            current_file_ = nullptr;
        }
        current_file_ = new ifstream();
        current_file_->exceptions(ifstream::failbit | ifstream::badbit | ifstream::eofbit);
        snprintf(string_buffer_, STRING_LEN, "AGIPD-BIN-R0243-M%02d-S%03d.dat", config_obj_->module_id, file_index);
        bf::path file_path(config_obj_->data_dir);
        file_path /= string_buffer_;
        try {
            current_file_->open(file_path.c_str(), std::ios::in | std::ios::binary);
            LOG4CXX_INFO(logger_, "successfully opened raw data file " << file_path.c_str());
            current_file_index_ = file_index;
            current_file_path_ = file_path.c_str();
        } catch (std::exception& e) {
            LOG4CXX_WARN(logger_, "failed to open file " << file_path.c_str() << " with error: " << strerror(errno));
            delete current_file_;
            current_file_ = nullptr;
            current_file_index_ = -1;
            return false;
        }
    }
    // read one data frame and send
    if (current_file_->tellg() != file_offset) {
        current_file_->seekg(file_offset);
    }
    bool succ_read = true;
    try {
        current_file_->read(frame_buffer_, FRAME_SIZE);
        if (current_file_->gcount() != FRAME_SIZE) {
            LOG4CXX_WARN(logger_, "read partial frame data from file " << current_file_path_);
            succ_read = false;
        }
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "failed read file " << current_file_path_ << " with error: " << strerror(errno));
        succ_read = false;
    }
    if (succ_read) {
        uint64_t key = gDC.decode_byte<uint64_t>(frame_buffer_, 12, 19);
        if (key != event_index) {
            LOG4CXX_WARN(logger_, "event_index " << event_index << " does not match with " << key << ".");
            return false;
        }
        if (!send_one_(head_buffer_, HEAD_SIZE, frame_buffer_, FRAME_SIZE)) {
            close_connection();
            LOG4CXX_WARN(logger_, "failed to send frame data.");
            return false;
        } else {
            LOG4CXX_DEBUG(logger_, "successfully send one frame of index " << event_index);
            return true;
        }
    } else {
        current_file_->close();
        delete current_file_;
        current_file_ = nullptr;
        current_file_index_ = 0;
        return false;
    }
}