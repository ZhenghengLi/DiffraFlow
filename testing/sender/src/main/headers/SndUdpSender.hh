#ifndef __SndUdpSender_H__
#define __SndUdpSender_H__

#include <log4cxx/logger.h>

namespace diffraflow {
    class SndUdpSender {
    public:
        SndUdpSender();
        ~SndUdpSender();

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif