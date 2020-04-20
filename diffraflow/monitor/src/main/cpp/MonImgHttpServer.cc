#include "MonImgHttpServer.hh"
#include "MonConfig.hh"
#include "ImageWithFeature.hh"
#include "ImageAnalysisResult.hh"

#include <fstream>
#include <msgpack.hpp>
#include <regex>
#include <chrono>

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::ifstream;
using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::MonImgHttpServer::logger_
    = log4cxx::Logger::getLogger("MonImgHttpServer");

diffraflow::MonImgHttpServer::MonImgHttpServer(MonConfig* conf_obj) {
    listener_ = nullptr;
    current_index_ = 0;
    config_obj_ = conf_obj;
    server_status_ = kNotStart;
}

diffraflow::MonImgHttpServer::~MonImgHttpServer() {

}

bool diffraflow::MonImgHttpServer::start(string host, int port) {
    if (server_status_ == kRunning) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&MonImgHttpServer::handleGet_, this, std::placeholders::_1));

    try {
        listener_->open().wait();
        server_status_ = kRunning;
        return true;
    } catch (std::exception& e) {
        LOG4CXX_ERROR(logger_, "failed to start http server: " << e.what());
        return false;
    } catch(...) {
        LOG4CXX_ERROR(logger_, "failed to start http server with unknown error.");
        return false;
    }
    return true;
}

void diffraflow::MonImgHttpServer::stop() {
    if (listener_ == nullptr) return;

    try {
        listener_->close().wait();
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "exception found when closing http listener: " << e.what());
    } catch (...) {
        LOG4CXX_WARN(logger_, "an unknown exception found when closing http listener.");
    }

    delete listener_;
    listener_ = nullptr;

    server_status_ = kStopped;
    cv_status_.notify_all();

}

void diffraflow::MonImgHttpServer::wait() {
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk,
        [this]() {return server_status_ != kRunning;}
    );
}

bool diffraflow::MonImgHttpServer::create_ingester_clients(const char* filename, int timeout) {
    ingester_clients_vec_.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        LOG4CXX_ERROR(logger_, "address file open failed.");
        return false;
    }
    http_client_config http_cc;
    http_cc.set_timeout(std::chrono::milliseconds(timeout));
    string oneline;
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments and empty lines
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // construct http client
        try {
            ingester_clients_vec_.push_back(http_client(uri(oneline), http_cc));
        } catch (std::exception& e) {
            LOG4CXX_ERROR(logger_, "failed to construct http client for ingester address "
                << oneline << " with exception: " << e.what());
            return false;
        }
    }
    if (ingester_clients_vec_.size() > 0) {
        return true;
    } else {
            LOG4CXX_ERROR(logger_, "empty address file: " << filename);
        return false;
    }
    return true;
}

bool diffraflow::MonImgHttpServer::request_one_image_(const string event_time_string,
    ImageWithFeature& image_with_feature, string& ingester_id_str) {

    lock_guard<mutex> lg(mtx_address_);

    if (ingester_clients_vec_.empty()) {
        LOG4CXX_WARN(logger_, "empty ingester clients list.");
        return false;
    }

    for (size_t addr_idx = current_index_; true; ) {
        http_response response = ingester_clients_vec_[addr_idx++].request(methods::GET, event_time_string).get();
        if (addr_idx >= ingester_clients_vec_.size()) {
            addr_idx = 0;
        }
        if (response.status_code() == status_codes::OK) {// succ
            if (response.headers().has("Ingester-ID")) {
                ingester_id_str = response.headers()["Ingester-ID"];
            } else {
                LOG4CXX_WARN(logger_, "no Ingester-ID in http response header.");
                return false;
            }
            vector<unsigned char> body_vec = response.extract_vector().get();
            try {
                msgpack::unpack((const char*) body_vec.data(), body_vec.size()).get().convert(image_with_feature);
            } catch (std::exception& e) {
                LOG4CXX_WARN(logger_, "failed to deserialize image_with_feature data with exception: " << e.what());
                return false;
            }
            current_index_ = addr_idx;
            return true;
        } else if (addr_idx == current_index_) {
            return false;
        }
    }

}

void diffraflow::MonImgHttpServer::do_analysis_(const ImageWithFeature& image_with_feature,
    ImageAnalysisResult& image_analysis_result) {
    image_analysis_result.image_with_feature = image_with_feature;
    // do some heavy analysis here and save result into image_analysis_result

}

void diffraflow::MonImgHttpServer::handleGet_(http_request message) {

    string relative_path = uri::decode(message.relative_uri().path());

    std::regex req_time_regex("^/(\\d+)$");
    std::smatch match_res;

    msgpack::sbuffer image_sbuff;
    http_response response;
    vector<unsigned char> response_data_vec;

    string event_time_string;
    if (relative_path == "/") {
        event_time_string = "";
    } else if (std::regex_match(relative_path, match_res, req_time_regex)) {
        event_time_string = match_res[1].str();
    } else {
        message.reply(status_codes::NotFound);
        return;
    }

    ImageWithFeature image_with_feature;
    string ingester_id_str;
    string monitor_id_str = std::to_string(config_obj_->monitor_id);
    if (request_one_image_(event_time_string, image_with_feature, ingester_id_str)) {
        string event_time_str = std::to_string(image_with_feature.image_data_raw.event_time);
        ImageAnalysisResult image_analysis_result;
        do_analysis_(image_with_feature, image_analysis_result);
        msgpack::pack(image_sbuff, image_analysis_result);
        response_data_vec.assign(image_sbuff.data(), image_sbuff.data() + image_sbuff.size());
        response.set_body(response_data_vec);
        response.set_status_code(status_codes::OK);
        response.headers().set_content_type("application/msgpack");
        response.headers().add(U("Monitor-ID"), monitor_id_str);
        response.headers().add(U("Ingester-ID"), ingester_id_str);
        response.headers().add(U("Event-Time"), event_time_str);
        response.headers().add(U("Cpp-Class"), U("diffraflow::ImageAnalysisResult"));
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        message.reply(response);
    } else {
        message.reply(status_codes::NotFound);
    }

}
