#include "CmbImgCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <boost/log/trivial.hpp>

diffraflow::CmbImgCache::CmbImgCache(size_t num_of_dets) {
    imgfrm_queues_len_ = num_of_dets;
    imgfrm_queues_arr_ = new queue<ImageFrame>[imgfrm_queues_len_];
    imgdat_queue_.set_maxsize(100);
}

diffraflow::CmbImgCache::~CmbImgCache() {

}

void diffraflow::CmbImgCache::put_frame(const ImageFrame& image_frame) {
    // in this function, put image from into priority queue, then try to do time alignment
    if (image_frame.det_id > imgfrm_queues_len_) {
        BOOST_LOG_TRIVIAL(warning) << "Detector ID is out of range: " << image_frame.det_id;
        return;
    }
    imgfrm_queues_arr_[image_frame.det_id].push(image_frame);
    if (do_alignment()) {
        // for debug only
        BOOST_LOG_TRIVIAL(info) << "Successfully do one alignment.";
    }
}

bool diffraflow::CmbImgCache::do_alignment() {
    // in this function, try to do time alignment, and put the aligned full image data into imgdat_queue_
    if (imgfrm_queues_arr_[0].empty()) {
        return false;
    }
    ImageData image_data;
    image_data.put_imgfrm(0, imgfrm_queues_arr_[0].front());
    BOOST_LOG_TRIVIAL(info) << "after put_imgfrm.";
    imgfrm_queues_arr_[0].pop();
    image_data.print();
    // imgdat_queue_.push(image_data);
    return true;
}

bool diffraflow::CmbImgCache::take_one_image(ImageData& image_data) {
    return imgdat_queue_.take(image_data);
}

void diffraflow::CmbImgCache::img_queue_stop() {
    imgdat_queue_.stop();
}

bool diffraflow::CmbImgCache::img_queue_stopped() {
    return imgdat_queue_.stopped();
}
