#include "MonImgHttpServer.hh"
#include "MonConfig.hh"
#include "ImageVisObject.hh"

#include <fstream>
#include <msgpack.hpp>
#include <regex>
#include <chrono>
#include <cstdlib>
#include <boost/algorithm/string.hpp>
#include <cpprest/rawptrstream.h>

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::ifstream;
using std::lock_guard;
using std::unique_lock;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::MonImgHttpServer::logger_ = log4cxx::Logger::getLogger("MonImgHttpServer");

diffraflow::MonImgHttpServer::MonImgHttpServer(MonConfig* conf_obj) {
    listener_ = nullptr;
    current_index_ = 0;
    config_obj_ = conf_obj;
    server_status_ = kNotStart;

    metrics.total_request_counts = 0;
    metrics.total_sent_counts = 0;
}

diffraflow::MonImgHttpServer::~MonImgHttpServer() {}

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
    } catch (...) {
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
    cv_status_.wait(ulk, [this]() { return server_status_ != kRunning; });
}

bool diffraflow::MonImgHttpServer::create_ingester_clients(const char* filename, int timeout) {
    ingester_clients_vec_.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        LOG4CXX_ERROR(logger_, "address file open failed.");
        return false;
    }
    http_client_config client_config;
    client_config.set_timeout(std::chrono::milliseconds(timeout));
    string oneline;
    const char* node_name_cstr = getenv("NODE_NAME");
    const char* node_ip_cstr = getenv("NODE_IP");
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments and empty lines
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // replace NODE_NAME or NODE_IP
        if (regex_match(oneline, regex(".*NODE_NAME.*")) && node_name_cstr != NULL) {
            oneline = regex_replace(oneline, regex("NODE_NAME"), node_name_cstr);
        } else if (regex_match(oneline, regex(".*NODE_IP.*")) && node_ip_cstr != NULL) {
            oneline = regex_replace(oneline, regex("NODE_IP"), node_ip_cstr);
        }
        // construct http client
        try {
            uri uri_val(oneline);
            http_client client(uri_val, client_config);
            ingester_clients_vec_.push_back(client);
            LOG4CXX_INFO(logger_, "created ingester client for " << uri_val.to_string());
        } catch (std::exception& e) {
            LOG4CXX_ERROR(logger_, "exception found when creating ingester client for " << oneline << ": " << e.what());
            return false;
        }
    }
    if (ingester_clients_vec_.size() > 0) {
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "no valid ingester addresses found in file: " << filename);
        return false;
    }
}

bool diffraflow::MonImgHttpServer::request_one_image_(
    const string key_string, ImageDataFeature& image_data_feature, string& ingester_id_str) {

    lock_guard<mutex> lg(mtx_client_);

    if (ingester_clients_vec_.empty()) {
        LOG4CXX_WARN(logger_, "empty ingester clients list.");
        return false;
    }

    for (size_t addr_idx = current_index_; true;) {
        http_response response;
        bool found_exception = false;

        LOG4CXX_DEBUG(
            logger_, "requesting data from \"" << ingester_clients_vec_[addr_idx].base_uri().to_string() << "\" ...");

        try {
            response = ingester_clients_vec_[addr_idx].request(methods::GET, key_string).get();
        } catch (std::exception& e) {
            found_exception = true;
            LOG4CXX_WARN(logger_, "exception found when requesting data from \""
                                      << ingester_clients_vec_[addr_idx].base_uri().to_string() << "\": " << e.what());
        }
        addr_idx++;
        if (addr_idx >= ingester_clients_vec_.size()) {
            addr_idx = 0;
        }
        if (!found_exception && response.status_code() == status_codes::OK) { // succ
            if (response.headers().has("Ingester-ID")) {
                ingester_id_str = response.headers()["Ingester-ID"];
            } else {
                LOG4CXX_WARN(logger_, "no Ingester-ID in http response header.");
                return false;
            }

            LOG4CXX_DEBUG(logger_, "received one image_data_feature from ingester " << ingester_id_str);

            vector<unsigned char> body_vec = response.extract_vector().get();

            LOG4CXX_DEBUG(logger_, "successfully extracted data of image_data_feature with size " << body_vec.size());

            try {
                msgpack::unpack((const char*)body_vec.data(), body_vec.size()).get().convert(image_data_feature);
            } catch (std::exception& e) {
                LOG4CXX_WARN(logger_, "failed to deserialize image_data_feature data with exception: " << e.what());
                return false;
            }

            LOG4CXX_DEBUG(logger_, "successfully unpacked image_data_feature");

            current_index_ = addr_idx;
            return true;
        } else if (addr_idx == current_index_) {

            LOG4CXX_DEBUG(logger_, "failed to get one image_data_feature currently in this node.");

            return false;
        } else {
            LOG4CXX_DEBUG(logger_, "failed to get one image_data_feature from this ingester, try next one.")
        }
    }
}

void diffraflow::MonImgHttpServer::do_analysis_(
    const ImageDataFeature& image_data_feature, ImageAnalysisResult& image_analysis_result) {
    // do some heavy analysis here and save result into image_analysis_result
    image_analysis_result.int_result = 123;
    image_analysis_result.float_result = 456;
}

void diffraflow::MonImgHttpServer::handleGet_(http_request message) {

    metrics.total_request_counts++;

    string relative_path = uri::decode(message.relative_uri().path());

    LOG4CXX_DEBUG(logger_, "get one request with path " << relative_path);

    std::regex req_regex("^/(\\d+)$");
    std::smatch match_res;

    msgpack::sbuffer image_sbuff;
    http_response response;
    response.headers().add(U("Access-Control-Allow-Origin"), U("*"));

    string key_string;
    if (relative_path == "/") {
        key_string = "";
    } else if (std::regex_match(relative_path, match_res, req_regex)) {
        key_string = match_res[1].str();
    } else {
        response.set_status_code(status_codes::NotFound);
        message.reply(response).get();
        return;
    }

    ImageDataFeature image_data_feature;
    string ingester_id_str;
    string monitor_id_str = std::to_string(config_obj_->monitor_id);

    if (request_one_image_(key_string, image_data_feature, ingester_id_str)) {
        if (!image_data_feature.image_data || !image_data_feature.image_feature) {
            LOG4CXX_WARN(logger_, "found unexpected null image_data or image_feature.");
            response.set_status_code(status_codes::InternalError);
            message.reply(response).get();
            return;
        }
        string key_str = std::to_string(image_data_feature.image_data->get_key());

        LOG4CXX_DEBUG(logger_, "successfully get one image_data_feature with key " << key_str);

        ImageVisObject image_vis_object;
        image_vis_object.image_data = make_shared<ImageDataSmall>(
            *image_data_feature.image_data, config_obj_->get_dy_energy_down_cut(), config_obj_->get_dy_energy_up_cut());
        image_vis_object.image_feature = image_data_feature.image_feature;
        image_vis_object.analysis_result = make_shared<ImageAnalysisResult>();
        do_analysis_(image_data_feature, *image_vis_object.analysis_result);

        LOG4CXX_DEBUG(logger_, "finished analysis for the image_data_feature.");

        msgpack::pack(image_sbuff, image_vis_object);
        concurrency::streams::istream data_stream = concurrency::streams::rawptr_stream<uint8_t>::open_istream(
            (const uint8_t*)image_sbuff.data(), image_sbuff.size());
        response.set_body(data_stream);

        response.set_status_code(status_codes::OK);
        response.headers().set_content_type("application/msgpack");
        response.headers().add(U("Node-Name"), config_obj_->node_name);
        response.headers().add(U("Monitor-ID"), monitor_id_str);
        response.headers().add(U("Ingester-ID"), ingester_id_str);
        response.headers().add(U("Event-Key"), key_str);
        response.headers().add(U("Cpp-Class"), U("diffraflow::ImageVisObject"));
        response.headers().add(U("Access-Control-Expose-Headers"), U("*"));

        message.reply(response).get();

        LOG4CXX_DEBUG(logger_, "the image data and analsyis result have been sent.");

        metrics.total_sent_counts++;
    } else {
        response.set_status_code(status_codes::NotFound);
        message.reply(response).get();
    }
}

json::value diffraflow::MonImgHttpServer::collect_metrics() {
    json::value root_json;
    root_json["total_request_counts"] = json::value::number(metrics.total_request_counts.load());
    root_json["total_sent_counts"] = json::value::number(metrics.total_sent_counts.load());
    return root_json;
}
