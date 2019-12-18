#ifndef IngImgWriter_H
#define IngImgWriter_H

#include <vector>
#include <mutex>
#include "BlockingQueue.hh"

using std::vector;
using std::mutex;
using std::lock_guard;

namespace diffraflow {

    class ImageData;

    class IngImgWriter {
    private:
        mutex mtx_;

        BlockingQueue<ImageData>* img_q_arr_;
        size_t img_q_size_;

        int wait_time_;
        bool stop_flag_;

    public:
        IngImgWriter(BlockingQueue<ImageData>* q_arr, size_t q_size);
        ~IngImgWriter();

        bool openfiles(const vector<int>& partitions);
        void closefiles();

        void set_stop();
        void run();

    };
}

#endif
