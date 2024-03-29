#include "DspImgFrmRecv.hh"
#include "ImageFrameRaw.hh"
#include "Decoder.hh"
#include "DspSender.hh"

#include "ImageFrameDgram.hh"

using std::make_shared;

log4cxx::LoggerPtr diffraflow::DspImgFrmRecv::logger_ = log4cxx::Logger::getLogger("DspImgFrmRecv");

diffraflow::DspImgFrmRecv::DspImgFrmRecv(
    string host, int port, DspSender** sender_arr, size_t sender_cnt, int rcvbufsize, int dgram_queue_size)
    : GenericDgramReceiver(host, port, rcvbufsize) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
    imgfrm_queue_.set_maxsize(dgram_queue_size);
    checker_thread_ = nullptr;
    // init metrics
    frame_metrics.total_received_count = 0;
    frame_metrics.total_checked_count = 0;
}

diffraflow::DspImgFrmRecv::~DspImgFrmRecv() { stop_checker(); }

void diffraflow::DspImgFrmRecv::set_max_queue_size(size_t max_qs) { imgfrm_queue_.set_maxsize(max_qs); }

void diffraflow::DspImgFrmRecv::process_datagram_(shared_ptr<ByteBuffer>& datagram) {

    if (datagram->size() < 100) {
        return;
    }
    uint8_t dgram_mod_id = gDC.decode_byte<uint8_t>(datagram->data(), 0, 0);
    uint16_t dgram_frm_sn = gDC.decode_byte<uint16_t>(datagram->data(), 1, 2);
    uint8_t dgram_seg_sn = gDC.decode_byte<uint8_t>(datagram->data(), 3, 3);

    LOG4CXX_DEBUG(logger_, "received one datagram: (mod_id, frm_sn, seg_sn, size) = ("
                               << (int)dgram_mod_id << ", " << (int)dgram_frm_sn << ", " << (int)dgram_seg_sn << ", "
                               << datagram->size() << ")");

    if (dgram_mod_id >= MOD_CNT) {
        return;
    }

    shared_ptr<ImageFrameRaw>& image_frame = image_frame_arr_[dgram_mod_id];

    if (dgram_seg_sn == 0) {
        if (image_frame) {
            imgfrm_queue_.push(image_frame);
            frame_metrics.total_received_count++;
        }
        image_frame = make_shared<ImageFrameRaw>();
        image_frame->add_dgram(datagram);
    } else {
        if (image_frame) {
            if (dgram_frm_sn == image_frame->dgram_frm_sn) {
                image_frame->add_dgram(datagram);
                if (image_frame->get_dgram_count() == BODY_COUNT + 1) {
                    imgfrm_queue_.push(image_frame);
                    image_frame = nullptr;
                    frame_metrics.total_received_count++;
                }
            } else {
                imgfrm_queue_.push(image_frame);
                image_frame = nullptr;
                frame_metrics.total_received_count++;
            }
        }
    }
}

void diffraflow::DspImgFrmRecv::start_checker(cpu_set_t* cpuset) {
    if (checker_thread_ != nullptr) {
        return;
    }
    checker_thread_ = new thread([this]() {
        shared_ptr<ImageFrameRaw> image_frame;
        while (imgfrm_queue_.take(image_frame)) {
            int check_res = image_frame->check_dgrams_integrity();
            if (check_res > 0) {
                frame_metrics.total_checked_count++;
                LOG4CXX_DEBUG(logger_, "successfully checked one image frame with size: " << check_res);
                size_t index = hash_long_(image_frame->get_key()) % sender_count_;
                if (sender_array_[index]->push(image_frame)) {
                    LOG4CXX_DEBUG(logger_, "pushed one image frame into sender[" << index << "].");
                } else {
                    LOG4CXX_WARN(logger_, "unable to push one image frame into sender[" << index << "].");
                }
            } else {
                LOG4CXX_DEBUG(logger_, "received bad image frame with checking error code: " << check_res);
            }
        }
    });
    if (cpuset && CPU_COUNT(cpuset) > 0) {
        int rc = pthread_setaffinity_np(checker_thread_->native_handle(), sizeof(cpu_set_t), cpuset);
        if (rc != 0) {
            LOG4CXX_WARN(logger_, "Failed to pthread_setaffinity_np for checker thread with error code: " << rc);
        }
    }
}

void diffraflow::DspImgFrmRecv::stop_checker() {
    if (checker_thread_ == nullptr) {
        return;
    }
    imgfrm_queue_.stop();
    checker_thread_->join();
    delete checker_thread_;
    checker_thread_ = nullptr;
}

uint32_t diffraflow::DspImgFrmRecv::hash_long_(uint64_t value) { return (uint32_t)(value ^ (value >> 32)); }

json::value diffraflow::DspImgFrmRecv::collect_metrics() {

    json::value root_json = GenericDgramReceiver::collect_metrics();

    json::value frame_metrics_json;
    frame_metrics_json["total_received_count"] = json::value::number(frame_metrics.total_received_count.load());
    frame_metrics_json["total_checked_count"] = json::value::number(frame_metrics.total_checked_count.load());
    frame_metrics_json["current_frame_queue_size"] = json::value::number((uint64_t)imgfrm_queue_.size());

    root_json["frame_stats"] = frame_metrics_json;

    return root_json;
}
