#include "IngImgWriter.hh"
#include "ImageData.hh"

#include <iostream>

using std::cout;
using std::endl;

diffraflow::IngImgWriter::IngImgWriter(BlockingQueue<ImageData>* q_arr, size_t q_size) {
    img_q_arr_ = q_arr;
    img_q_size_ = q_size;
    wait_time_ = 100;
    stop_flag_ = false;
}

diffraflow::IngImgWriter::~IngImgWriter() {

}

bool diffraflow::IngImgWriter::openfiles(const vector<int>& partitions) {
    lock_guard<mutex> lk(mtx_);
    // open HDF5 files on DFS for each partition
    return true;
}

void diffraflow::IngImgWriter::closefiles() {
    lock_guard<mutex> lk(mtx_);

}

void diffraflow::IngImgWriter::set_stop() {
    lock_guard<mutex> lk(mtx_);
    stop_flag_ = true;
}

void diffraflow::IngImgWriter::run() {
    cout << "Start running ..." << endl;
    while (true) {
        bool is_stopped = false;
        {
            lock_guard<mutex> lk(mtx_);
            is_stopped = stop_flag_;
        }
        if (is_stopped) {
            break;
        }
        int written_count = 0;
        for (size_t i = 0; i < img_q_size_; i++) {
            ImageData img_data;
            if (img_q_arr_[i].get(img_data)) {
                // write img_data here
                written_count++;
            }
        } 
        if (written_count < 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_));
        }
    }
}
