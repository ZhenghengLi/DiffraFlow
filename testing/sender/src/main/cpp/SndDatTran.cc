#include "SndDatTran.hh"
#include "SndConfig.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "SndTcpSender.hh"
#include "SndUdpSender.hh"

#include <cstdio>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/filesystem.hpp>
#include <sched.h>

#define FRAME_SIZE 131096
#define STRING_LEN 512

namespace bf = boost::filesystem;
namespace bs = boost::system;

using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::SndDatTran::logger_ = log4cxx::Logger::getLogger("SndDatTran");

diffraflow::SndDatTran::SndDatTran(SndConfig* conf_obj) {
    config_obj_ = conf_obj;
    sender_type_ = kNotSet;
    tcp_sender_ = nullptr;
    udp_sender_ = nullptr;
    frame_buffer_ = new char[FRAME_SIZE];
    string_buffer_ = new char[STRING_LEN];
    current_file_ = nullptr;
    current_file_index_ = -1;

    event_queue_.set_maxsize(100);
    sender_thread_ = nullptr;

    transfer_metrics.invoke_counts = 0;
    transfer_metrics.busy_counts = 0;
    transfer_metrics.large_index_counts = 0;
    transfer_metrics.read_succ_counts = 0;
    transfer_metrics.key_match_counts = 0;
    transfer_metrics.send_succ_counts = 0;
    transfer_metrics.send_fail_counts = 0;
    transfer_metrics.read_send_succ_counts = 0;
    transfer_metrics.read_send_fail_counts = 0;
}

diffraflow::SndDatTran::~SndDatTran() {
    delete[] frame_buffer_;
    delete[] string_buffer_;
    if (current_file_ != nullptr) {
        current_file_->close();
        delete current_file_;
        current_file_ = nullptr;
    }
    delete_sender();
}

bool diffraflow::SndDatTran::create_tcp_sender(string dispatcher_host, int dispatcher_port, uint32_t sender_id) {
    tcp_sender_ = new SndTcpSender(dispatcher_host, dispatcher_port, sender_id);
    if (tcp_sender_->connect_to_server()) {
        LOG4CXX_INFO(logger_, "successfully connected to dispatcher: " << tcp_sender_->get_server_address());
        sender_type_ = kTCP;
        return true;
    } else {
        LOG4CXX_WARN(logger_, "failed to connect to dispatcher: " << tcp_sender_->get_server_address());
        return false;
    }
}

bool diffraflow::SndDatTran::create_udp_sender(string dispatcher_host, int dispatcher_port) {
    udp_sender_ = new SndUdpSender();
    if (udp_sender_->init_addr_sock(dispatcher_host, dispatcher_port)) {
        LOG4CXX_INFO(
            logger_, "successfully initialized udp socket for dispatcher: " << udp_sender_->get_receiver_address());
        sender_type_ = kUDP;
        return true;
    } else {
        LOG4CXX_WARN(
            logger_, "failed to initialize udp socket for dispatcher: " << udp_sender_->get_receiver_address());
        return false;
    }
}

void diffraflow::SndDatTran::delete_sender() {
    if (tcp_sender_ != nullptr) {
        tcp_sender_->close_connection();
        delete tcp_sender_;
        tcp_sender_ = nullptr;
    }
    if (udp_sender_ != nullptr) {
        udp_sender_->close_sock();
        delete udp_sender_;
        udp_sender_ = nullptr;
    }
    sender_type_ = kNotSet;
}

bool diffraflow::SndDatTran::push_event(uint32_t event_index) { return event_queue_.push(event_index); }

bool diffraflow::SndDatTran::start_sender(int cpu_id) {

    if (sender_thread_ != nullptr) {
        return true;
    }

    int num_cpus = std::thread::hardware_concurrency();
    if (cpu_id >= num_cpus) {
        LOG4CXX_ERROR(logger_, "CPU id (" << cpu_id << ") is too large, it should be smaller than " << num_cpus);
        return false;
    }

    event_queue_.resume();
    sender_thread_ = new thread([this] {
        uint32_t event_index = 0;
        while (event_queue_.take(event_index)) {
            if (read_and_send(event_index)) {
                transfer_metrics.read_send_succ_counts++;
                LOG4CXX_DEBUG(logger_, "successfully sent event " << event_index);
            } else {
                transfer_metrics.read_send_fail_counts++;
                LOG4CXX_WARN(logger_, "failed to send event " << event_index);
            }
        }
    });

    // bind cpu
    bool cpu_bind_succ = true;
    if (cpu_id >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        int rc = pthread_setaffinity_np(sender_thread_->native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc == 0) {
            LOG4CXX_INFO(logger_, "successfully bind sending thread on cpu " << cpu_id);
            cpu_bind_succ = true;
        } else {
            LOG4CXX_ERROR(logger_, "error calling pthread_setaffinity_np with error number: " << rc);
            cpu_bind_succ = false;
        }
    }
    if (cpu_bind_succ) {
        return true;
    } else {
        stop_sender();
        return false;
    }
}

void diffraflow::SndDatTran::stop_sender() {
    if (sender_thread_ != nullptr) {
        event_queue_.stop();
        sender_thread_->join();
        delete sender_thread_;
        sender_thread_ = nullptr;
    }
}

bool diffraflow::SndDatTran::read_and_send(uint32_t event_index) {

    transfer_metrics.invoke_counts++;

    if (sender_type_ == kNotSet) {
        LOG4CXX_WARN(logger_, "TCP or UDP sender is not created.");
        return false;
    }

    unique_lock<mutex> data_lk(data_mtx_, std::try_to_lock);
    if (!data_lk.owns_lock()) {
        LOG4CXX_WARN(
            logger_, "data transfer of a previous event is on going, and event " << event_index << " is jumped.");
        transfer_metrics.busy_counts++;
        return false;
    }

    if (event_index >= config_obj_->total_events) {
        LOG4CXX_WARN(
            logger_, "event index " << event_index << " is larger than total events" << config_obj_->total_events);
        transfer_metrics.large_index_counts++;
        return false;
    }

    int file_index = event_index / config_obj_->events_per_file;
    int64_t file_offset = event_index % config_obj_->events_per_file;
    file_offset *= FRAME_SIZE;
    // open file if necessary
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
    // seek position if necessary
    if (current_file_->tellg() != file_offset) {
        current_file_->seekg(file_offset);
    }
    // read one data frame
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
    // send the data frame if read succeeds
    if (succ_read) {
        transfer_metrics.read_succ_counts++;
        uint64_t key = gDC.decode_byte<uint64_t>(frame_buffer_, 12, 19);
        if (key == event_index) {
            transfer_metrics.key_match_counts++;
        } else {
            LOG4CXX_WARN(logger_, "event_index " << event_index << " does not match with " << key << ".");
            return false;
        }

        if (sender_type_ == kTCP) {
            if (tcp_sender_->send_frame(frame_buffer_, FRAME_SIZE)) {
                LOG4CXX_DEBUG(logger_, "tcp: successfully send one frame of index " << event_index);
                transfer_metrics.send_succ_counts++;
                return true;
            } else {
                tcp_sender_->close_connection();
                LOG4CXX_WARN(logger_, "tcp: failed to send frame data.");
                transfer_metrics.send_fail_counts++;
                return false;
            }
        } else if (sender_type_ == kUDP) {
            if (udp_sender_->send_frame(frame_buffer_, FRAME_SIZE)) {
                LOG4CXX_DEBUG(logger_, "udp: successfully send one frame of index " << event_index);
                transfer_metrics.send_succ_counts++;
                return true;
            } else {
                LOG4CXX_WARN(logger_, "udp: failed to send frame data.");
                transfer_metrics.send_fail_counts++;
                return false;
            }
        }
    } else {
        current_file_->close();
        delete current_file_;
        current_file_ = nullptr;
        current_file_index_ = 0;
        return false;
    }
}

json::value diffraflow::SndDatTran::collect_metrics() {

    json::value transfer_metrics_json;
    transfer_metrics_json["invoke_counts"] = json::value::number(transfer_metrics.invoke_counts);
    transfer_metrics_json["busy_counts"] = json::value::number(transfer_metrics.busy_counts);
    transfer_metrics_json["large_index_counts"] = json::value::number(transfer_metrics.large_index_counts);
    transfer_metrics_json["read_succ_counts"] = json::value::number(transfer_metrics.read_succ_counts);
    transfer_metrics_json["key_match_counts"] = json::value::number(transfer_metrics.key_match_counts);
    transfer_metrics_json["send_succ_counts"] = json::value::number(transfer_metrics.send_succ_counts);
    transfer_metrics_json["send_fail_counts"] = json::value::number(transfer_metrics.send_fail_counts);
    transfer_metrics_json["read_send_succ_counts"] = json::value::number(transfer_metrics.read_send_succ_counts);
    transfer_metrics_json["read_send_fail_counts"] = json::value::number(transfer_metrics.read_send_fail_counts);

    json::value root_json;
    root_json["transfer_stat"] = transfer_metrics_json;
    if (sender_type_ == kTCP) {
        root_json["tcp_sender_stat"] = tcp_sender_->collect_metrics();
    } else if (sender_type_ == kUDP) {
        root_json["udp_sender_stat"] = udp_sender_->collect_metrics();
    }

    return root_json;
}