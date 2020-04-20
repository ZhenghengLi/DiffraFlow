#include "IngImgHttpServer.hh"
#include "IngImageFilter.hh"
#include "ImageWithFeature.hh"

#include <msgpack.hpp>
#include <regex>

using namespace web;
using namespace http;
using namespace experimental::listener;

log4cxx::LoggerPtr diffraflow::IngImgHttpServer::logger_
    = log4cxx::Logger::getLogger("IngImgHttpServer");

diffraflow::IngImgHttpServer::IngImgHttpServer(IngImageFilter* img_filter, int ing_id) {
    image_filter_ = img_filter;
    listener_ = nullptr;
    ingester_id_ = ing_id;
}

diffraflow::IngImgHttpServer::~IngImgHttpServer() {
    stop();
}

bool diffraflow::IngImgHttpServer::start(string host, int port) {
    if (listener_ != nullptr) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&IngImgHttpServer::handleGet_, this, std::placeholders::_1));

    try {
        listener_->open().wait();
        return true;
    } catch (std::exception& e) {
        LOG4CXX_ERROR(logger_, "failed to start http server: " << e.what());
        return false;
    } catch(...) {
        LOG4CXX_ERROR(logger_, "failed to start http server with unknown error.");
        return false;
    }

}

void diffraflow::IngImgHttpServer::stop() {
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

}

void diffraflow::IngImgHttpServer::handleGet_(http_request message) {

    string relative_path = uri::decode(message.relative_uri().path());

    std::regex req_time_regex("^/(\\d+)$");
    std::smatch match_res;

    msgpack::sbuffer image_sbuff;
    http_response response;
    vector<unsigned char> response_data_vec;

    if (relative_path == "/") {
        ImageWithFeature current_image;
        if (!image_filter_->get_current_image(current_image)) {
            message.reply(status_codes::NotFound);
            return;
        }
        string event_time_str = std::to_string(current_image.image_data_raw.event_time);
        string ingester_id_str = std::to_string(ingester_id_);

        msgpack::pack(image_sbuff, current_image);
        response_data_vec.assign(image_sbuff.data(), image_sbuff.data() + image_sbuff.size());
        response.set_body(response_data_vec);
        response.set_status_code(status_codes::OK);
        response.headers().set_content_type("application/msgpack");
        response.headers().add(U("Ingester-ID"), ingester_id_str);
        response.headers().add(U("Event-Time"), event_time_str);
        response.headers().add(U("Cpp-Class"), U("diffraflow::ImageWithFeature"));
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        message.reply(response);

    } else if (std::regex_match(relative_path, match_res, req_time_regex)) {
        uint64_t request_time = std::stoul(match_res[1].str());
        ImageWithFeature current_image;
        if (!image_filter_->get_current_image(current_image)) {
            message.reply(status_codes::NotFound);
            return;
        }
        string event_time_str = std::to_string(current_image.image_data_raw.event_time);
        string ingester_id_str = std::to_string(ingester_id_);
        if (request_time < current_image.image_data_raw.event_time) {

            msgpack::pack(image_sbuff, current_image);
            response_data_vec.assign(image_sbuff.data(), image_sbuff.data() + image_sbuff.size());
            response.set_body(response_data_vec);
            response.set_status_code(status_codes::OK);
            response.headers().set_content_type("application/msgpack");
            response.headers().add(U("Ingester-ID"), ingester_id_str);
            response.headers().add(U("Event-Time"), event_time_str);
            response.headers().add(U("Cpp-Class"), U("diffraflow::ImageWithFeature"));
            response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
            message.reply(response);

        } else {
            message.reply(status_codes::NotFound);
        }

    } else {
        message.reply(status_codes::NotFound);
    }
}
