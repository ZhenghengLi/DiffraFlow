#include "MonImgHttpServer.hh"
#include <fstream>

#include <msgpack.hpp>
#include <regex>

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::ifstream;

log4cxx::LoggerPtr diffraflow::MonImgHttpServer::logger_
    = log4cxx::Logger::getLogger("MonImgHttpServer");

diffraflow::MonImgHttpServer::MonImgHttpServer() {
    listener_ = nullptr;
}

diffraflow::MonImgHttpServer::~MonImgHttpServer() {

}

bool diffraflow::MonImgHttpServer::start(string host, int port) {

    return true;
}

bool diffraflow::MonImgHttpServer::load_ingaddr_list(const char* filename) {
    ingester_addresses_vec_.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        LOG4CXX_ERROR(logger_, "address file open failed.");
        return false;
    }
    string oneline;
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments and empty lines
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // oneline should start with http://
        ingester_addresses_vec_.push_back(oneline);
    }
    if (ingester_addresses_vec_.size() > 0) {
        return true;
    } else {
            LOG4CXX_ERROR(logger_, "empty address file: " << filename);
        return false;
    }
    return true;
}