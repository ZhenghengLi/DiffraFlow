#include "AggHttpServer.hh"
#include "AggMetrics.hh"

#include <map>
#include <algorithm>

using std::map;
using std::pair;
using std::string;

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::AggHttpServer::logger_ = log4cxx::Logger::getLogger("AggHttpServer");

diffraflow::AggHttpServer::AggHttpServer(AggMetrics* metrics) {
    listener_ = nullptr;
    aggregated_metrics_ = metrics;
    server_status_ = kNotStart;
}

diffraflow::AggHttpServer::~AggHttpServer() {
    //
}

bool diffraflow::AggHttpServer::start(string host, int port) {
    if (server_status_ == kRunning) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }

    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&AggHttpServer::handleGet_, this, std::placeholders::_1));

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

void diffraflow::AggHttpServer::stop() {
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

void diffraflow::AggHttpServer::wait() {
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return server_status_ != kRunning; });
}

void diffraflow::AggHttpServer::handleGet_(http_request message) {

    http_response response;
    response.headers().add(U("Access-Control-Allow-Origin"), U("*"));

    if (message.relative_uri().path() != "/") {
        response.set_status_code(status_codes::NotFound);
        message.reply(response).get();
        return;
    }

    // bool compress_flag = false;
    // map<utility::string_t, utility::string_t> query_map = uri::split_query(message.relative_uri().query());
    // for (const pair<utility::string_t, utility::string_t>& item : query_map) {
    //     if (item.first == "compress" && item.second == "true") {
    //         compress_flag = true;
    //     }
    // }

    json::value root_json = aggregated_metrics_->get_metrics();
    response.set_body(root_json);
    response.set_status_code(status_codes::OK);
    message.reply(response).get();
}
