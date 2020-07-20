#ifndef __SndDatTran_H__
#define __SndDatTran_H__

#include <log4cxx/logger.h>

#include "GenericClient.hh"

namespace diffraflow {

    class SndConfig;

    class SndDatTran : public GenericClient {
    public:
        explicit SndDatTran(SndConfig* conf_obj);
        ~SndDatTran();

        bool read_and_send(uint32_t event_index);

    private:
        SndConfig* config_obj_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif