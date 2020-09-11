#include "DspImgFrmRecv.hh"
#include "ImageFrameRaw.hh"
#include "Decoder.hh"

using std::make_shared;

log4cxx::LoggerPtr diffraflow::DspImgFrmRecv::logger_ = log4cxx::Logger::getLogger("DspImgFrmRecv");

diffraflow::DspImgFrmRecv::DspImgFrmRecv(string host, int port) : GenericDgramReceiver(host, port) {
    //
}

diffraflow::DspImgFrmRecv::~DspImgFrmRecv() {
    //
}

void diffraflow::DspImgFrmRecv::process_datagram_(shared_ptr<vector<char>>& datagram) {
    if (datagram->size() < 100) {
        return;
    }
    uint8_t dgram_mod_id = gDC.decode_byte<uint8_t>(datagram->data(), 0, 0);
    if (dgram_mod_id >= MOD_CNT) {
        return;
    }
    shared_ptr<ImageFrameRaw>& image_frame = image_frame_arr[dgram_mod_id];
    uint16_t dgram_frm_sn = gDC.decode_byte<uint16_t>(datagram->data(), 1, 2);
    uint8_t dgram_seg_sn = gDC.decode_byte<uint8_t>(datagram->data(), 3, 3);
    if (dgram_seg_sn == 0) {
        if (image_frame) {
            // push image_frame in checking queue
        }
        image_frame = make_shared<ImageFrameRaw>();
        image_frame->add_dgram(datagram);
    } else {
        if (image_frame) {
            if (dgram_frm_sn == image_frame->dgram_frm_sn) {
                image_frame->add_dgram(datagram);
            } else {
                // push image_frame in checking queue
                image_frame = nullptr;
            }
        }
    }
}
