#include "IngImgHttpServer.hh"
#include "IngImageFilter.hh"
#include "ImageWithFeature.hh"
#include "ImageDataFeature.hh"

#include <msgpack.hpp>
#include <regex>
#include <memory>
#include <cpprest/rawptrstream.h>

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::vector;
using std::shared_ptr;

log4cxx::LoggerPtr diffraflow::IngImgHttpServer::logger_ = log4cxx::Logger::getLogger("IngImgHttpServer");

diffraflow::IngImgHttpServer::IngImgHttpServer(IngImageFilter* img_filter, int ing_id) {
    image_filter_ = img_filter;
    listener_ = nullptr;
    ingester_id_ = ing_id;

    metrics.total_request_counts = 0;
    metrics.total_sent_counts = 0;
}

diffraflow::IngImgHttpServer::~IngImgHttpServer() { stop(); }

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
    } catch (...) {
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

    metrics.total_request_counts++;

    string relative_path = uri::decode(message.relative_uri().path());

    std::regex req_regex("^/(\\d+)$");
    std::smatch match_res;

    // converting sequence: current_image => image_data_feature => image_sbuff
    ImageDataFeature image_data_feature;
    msgpack::sbuffer image_sbuff;
    http_response response;

    if (relative_path == "/") {
        shared_ptr<ImageWithFeature> current_image = image_filter_->get_current_image();
        if (current_image) {
            image_data_feature = *current_image;
        } else {
            message.reply(status_codes::NotFound).get();
            return;
        }
        string key_str = std::to_string(image_data_feature.image_data->get_key());
        string ingester_id_str = std::to_string(ingester_id_);

        msgpack::pack(image_sbuff, image_data_feature);
        concurrency::streams::istream data_stream = concurrency::streams::rawptr_stream<uint8_t>::open_istream(
            (const uint8_t*)image_sbuff.data(), image_sbuff.size());

        response.set_body(data_stream);

        response.set_status_code(status_codes::OK);
        response.headers().set_content_type("application/msgpack");
        response.headers().add(U("Ingester-ID"), ingester_id_str);
        response.headers().add(U("Event-Key"), key_str);
        response.headers().add(U("Cpp-Class"), U("diffraflow::ImageDataFeature"));
        response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
        response.headers().add(U("Access-Control-Expose-Headers"), U("*"));

        message.reply(response).get();

        metrics.total_sent_counts++;

    } else if (std::regex_match(relative_path, match_res, req_regex)) {
        uint64_t request_key = std::stoul(match_res[1].str());

        shared_ptr<ImageWithFeature> current_image = image_filter_->get_current_image();
        if (current_image) {
            image_data_feature = *current_image;
        } else {
            message.reply(status_codes::NotFound).get();
            return;
        }
        string key_str = std::to_string(image_data_feature.image_data->get_key());
        string ingester_id_str = std::to_string(ingester_id_);

        if (request_key < image_data_feature.image_data->get_key()) {

            msgpack::pack(image_sbuff, image_data_feature);
            concurrency::streams::istream data_stream = concurrency::streams::rawptr_stream<uint8_t>::open_istream(
                (const uint8_t*)image_sbuff.data(), image_sbuff.size());

            response.set_body(data_stream);

            response.set_status_code(status_codes::OK);
            response.headers().set_content_type("application/msgpack");
            response.headers().add(U("Ingester-ID"), ingester_id_str);
            response.headers().add(U("Event-Key"), key_str);
            response.headers().add(U("Cpp-Class"), U("diffraflow::ImageDataFeature"));
            response.headers().add(U("Access-Control-Allow-Origin"), U("*"));
            response.headers().add(U("Access-Control-Expose-Headers"), U("*"));

            message.reply(response).get();

            metrics.total_sent_counts++;

        } else {
            message.reply(status_codes::NotFound).get();
        }

    } else {
        message.reply(status_codes::NotFound).get();
    }
}

json::value diffraflow::IngImgHttpServer::collect_metrics() {
    json::value root_json;
    root_json["total_request_counts"] = json::value::number(metrics.total_request_counts.load());
    root_json["total_sent_counts"] = json::value::number(metrics.total_sent_counts.load());
    return root_json;
}
