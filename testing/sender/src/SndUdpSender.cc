#include "SndUdpSender.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include <cstring>
#include "ImageFrameDgram.hh"

log4cxx::LoggerPtr diffraflow::SndUdpSender::logger_ = log4cxx::Logger::getLogger("SndUdpSender");

diffraflow::SndUdpSender::SndUdpSender(int sndbufsize, int sender_port) : GenericDgramSender(sndbufsize) {
    set_sender_port(sender_port);
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
    memcpy(dgram_buffer_ + 4, buffer, HEAD_SIZE);
    if (send_datagram(dgram_buffer_, HEAD_SIZE + 4)) {
        LOG4CXX_DEBUG(logger_, "successfully sent datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                   << module_id << ", " << frame_sequence_number_ << ", "
                                   << (int)segment_sequence_number_ << ", " << HEAD_SIZE + 4 << ")");
    } else {
        succ_flag = false;
        LOG4CXX_WARN(logger_, "failed to send datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                  << module_id << ", " << frame_sequence_number_ << ", "
                                  << (int)segment_sequence_number_ << ", " << HEAD_SIZE + 4 << ")");
    }
    segment_sequence_number_++;

    // send subsequent segments
    for (size_t i = 0; succ_flag && i < BODY_COUNT; i++) {
        gPS.serializeValue<uint8_t>(segment_sequence_number_, dgram_buffer_ + 3, 1);
        size_t offset = HEAD_SIZE + i * BODY_SIZE;
        memcpy(dgram_buffer_ + 4, buffer + offset, BODY_SIZE);
        if (send_datagram(dgram_buffer_, BODY_SIZE + 4)) {
            LOG4CXX_DEBUG(logger_, "successfully sent datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                       << module_id << ", " << frame_sequence_number_ << ", "
                                       << (int)segment_sequence_number_ << ", " << BODY_SIZE + 4 << ")");
        } else {
            succ_flag = false;
            LOG4CXX_WARN(logger_, "failed to send datagram: (mod_id, frm_sn, seg_sn, size) = ("
                                      << module_id << ", " << frame_sequence_number_ << ", "
                                      << (int)segment_sequence_number_ << ", " << BODY_SIZE + 4 << ")");
        }
        segment_sequence_number_++;
    }

    frame_sequence_number_++;

    return succ_flag;
}
