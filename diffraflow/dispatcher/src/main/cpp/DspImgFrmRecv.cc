#include "DspImgFrmRecv.hh"

log4cxx::LoggerPtr diffraflow::DspImgFrmRecv::logger_ = log4cxx::Logger::getLogger("DspImgFrmRecv");

diffraflow::DspImgFrmRecv::DspImgFrmRecv(string host, int port) : GenericDgramReceiver(host, port) {
    //
}

diffraflow::DspImgFrmRecv::~DspImgFrmRecv() {
    //
}

void diffraflow::DspImgFrmRecv::process_datagram_(shared_ptr<vector<char>>& datagram) {
    //
}