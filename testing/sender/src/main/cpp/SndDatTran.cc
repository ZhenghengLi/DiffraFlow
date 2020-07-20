#include "SndDatTran.hh"
#include "SndConfig.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::SndDatTran::logger_ = log4cxx::Logger::getLogger("SndDatTran");

diffraflow::SndDatTran::SndDatTran(SndConfig* conf_obj)
    : GenericClient(conf_obj->dispatcher_host, conf_obj->dispatcher_port, conf_obj->sender_id, 0xFFDD1234, 0xFFF22DDD,
          0xDDD22FFF) {
    config_obj_ = conf_obj;
}

diffraflow::SndDatTran::~SndDatTran() {}

bool diffraflow::SndDatTran::read_and_send(uint32_t event_index) {
    if (event_index >= config_obj_->total_events) {
        return false;
    }
    // read one data frame and send
    return true;
}