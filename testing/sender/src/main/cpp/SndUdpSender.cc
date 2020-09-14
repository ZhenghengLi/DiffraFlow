#include "SndUdpSender.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include <cstring>

#define DGRAM_MSIZE 1500
#define FRAME_SIZE 131096

log4cxx::LoggerPtr diffraflow::SndUdpSender::logger_ = log4cxx::Logger::getLogger("SndUdpSender");

diffraflow::SndUdpSender::SndUdpSender() {
    dgram_buffer_ = new char[DGRAM_MSIZE];
    frame_sequence_number_ = 0;
    segment_sequence_number_ = 0;
}

diffraflow::SndUdpSender::~SndUdpSender() {
    delete[] dgram_buffer_;
    dgram_buffer_ = nullptr;
}

bool diffraflow::SndUdpSender::send_frame(const char* buffer, size_t len) {
    if (len != FRAME_SIZE) {
        LOG4CXX_WARN(logger_, "wrong frame size: " << len);
        return false;
    }
    uint16_t module_id = gDC.decode_byte<uint16_t>(buffer, 6, 7);
    if (module_id > 15) {
        LOG4CXX_WARN(logger_, "module_id is too large: " << module_id);
        return false;
    }
    segment_sequence_number_ = 0;

    bool succ_flag = true;

    // send the first segment
    gPS.serializeValue<uint8_t>((uint8_t)module_id, dgram_buffer_, 1);
    gPS.serializeValue<uint16_t>(frame_sequence_number_, dgram_buffer_ + 1, 2);
    gPS.serializeValue<uint8_t>(segment_sequence_number_, dgram_buffer_ + 3, 1);
    segment_sequence_number_++;
    memcpy(dgram_buffer_ + 4, buffer, 1376);
    if (send_datagram(dgram_buffer_, 1380)) {
        LOG4CXX_DEBUG(logger_, "successfully sent datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                   << module_id << ", " << frame_sequence_number_ << ", "
                                   << (int)segment_sequence_number_ << ", " << 1380 << ")");
    } else {
        succ_flag = false;
        LOG4CXX_WARN(logger_, "failed to send datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                  << module_id << ", " << frame_sequence_number_ << ", "
                                  << (int)segment_sequence_number_ << ", " << 1380 << ")");
    }

    // send subsequent segments
    for (size_t i = 0; succ_flag && i < 94; i++) {
        gPS.serializeValue<uint8_t>(segment_sequence_number_, dgram_buffer_ + 3, 1);
        segment_sequence_number_++;
        size_t offset = 1376 + i * 1380;
        memcpy(dgram_buffer_ + 4, buffer + offset, 1380);
        if (send_datagram(dgram_buffer_, 1384)) {
            LOG4CXX_DEBUG(logger_, "successfully sent datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                       << module_id << ", " << frame_sequence_number_ << ", "
                                       << (int)segment_sequence_number_ << ", " << 1384 << ")");

        } else {
            succ_flag = false;
            LOG4CXX_WARN(logger_, "failed to send datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                      << module_id << ", " << frame_sequence_number_ << ", "
                                      << (int)segment_sequence_number_ << ", " << 1384 << ")");
        }
    }

    frame_sequence_number_++;

    return succ_flag;
}
