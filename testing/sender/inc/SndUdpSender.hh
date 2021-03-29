#ifndef __SndUdpSender_H__
#define __SndUdpSender_H__

#include "GenericDgramSender.hh"
#include <log4cxx/logger.h>

namespace diffraflow {
    class SndUdpSender : public GenericDgramSender {
    public:
        explicit SndUdpSender(int sndbufsize = 4 * 1024 * 1024, int sender_port = -1);
        ~SndUdpSender();

        bool send_frame(const char* buffer, size_t len);

    private:
        char* dgram_buffer_;
        uint16_t frame_sequence_number_;
        uint8_t segment_sequence_number_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif