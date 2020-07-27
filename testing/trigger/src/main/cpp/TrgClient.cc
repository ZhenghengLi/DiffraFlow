#include "TrgClient.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::unique_lock;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::TrgClient::logger_ = log4cxx::Logger::getLogger("TrgClient");

diffraflow::TrgClient::TrgClient(string sender_host, int sender_port, uint32_t trigger_id)
    : GenericClient(sender_host, sender_port, trigger_id, 0xBBFF1234, 0xBBB22FFF, 0xFFF22BBB) {
    buffer_ = new char[4];

    worker_ = nullptr;
    running_flag_ = true;
    trigger_flag_ = false;
    target_event_index_ = 0;
}

diffraflow::TrgClient::~TrgClient() {
    stop();
    delete[] buffer_;
}

bool diffraflow::TrgClient::start() {
    // connect to server
    if (connect_to_server()) {
        LOG4CXX_INFO(logger_, "successfully connected to server " << get_server_address() << ".");
    } else {
        LOG4CXX_ERROR(logger_, "failed to connect to server " << get_server_address() << ".");
        return false;
    }
    // start worker thread
    worker_ = new thread([this]() {
        unique_lock<mutex> ulk(trigger_mtx_);
        while (true) {
            trigger_cv_.wait(ulk, [this]() { return !running_flag_ || trigger_flag_; });
            if (!running_flag_) break;
            success_ = send_event_index_(target_event_index_);
            trigger_flag_ = false;
            wait_cv_.notify_all();
        }
    });
}

void diffraflow::TrgClient::stop() {
    {
        lock_guard<mutex> lg(trigger_mtx_);
        running_flag_ = false;
        trigger_cv_.notify_all();
    }
    if (worker_ != nullptr) {
        worker_->join();
        delete worker_;
        worker_ = nullptr;
    }
}

bool diffraflow::TrgClient::trigger(const uint32_t event_index) {
    if (trigger_flag_) return false;
    lock_guard<mutex> lg(trigger_mtx_);
    target_event_index_ = event_index;
    trigger_flag_ = true;
    trigger_cv_.notify_all();
    return true;
}

bool diffraflow::TrgClient::wait() {
    unique_lock<mutex> ulk(trigger_mtx_);
    wait_cv_.wait(ulk, [this]() { return !running_flag_ || !trigger_flag_; });
    if (!running_flag_) return false;
    return success_;
}

bool diffraflow::TrgClient::send_event_index_(const uint32_t event_index) {
    LOG4CXX_DEBUG(logger_, get_server_address() << "  triggering " << event_index << " ...");
    if (not_connected()) {
        LOG4CXX_INFO(logger_, "connection to sender is lost, try to reconnect.");
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to sender.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to sender.");
            return false;
        }
    }

    gPS.serializeValue<uint32_t>(event_index, buffer_, 4);

    if (send_one_(buffer_, 4, nullptr, 0)) {
        LOG4CXX_INFO(logger_, "successfully sent event index " << event_index << " to " << get_server_address());
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "failed to send event index " << event_index << " to " << get_server_address());
        close_connection();
        return false;
    }
}
