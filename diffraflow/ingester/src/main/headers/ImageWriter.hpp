#ifndef ImageWriter_H
#define ImageWriter_H

#include <vector>
#include <mutex>
#include "BlockingQueue.hpp"

using std::vector;
using std::mutex;
using std::lock_guard;

namespace shine {

    class ImageData;

    class ImageWriter {
    private:
        mutex mtx_;

        BlockingQueue<ImageData>* img_q_arr_;
        size_t img_q_size_;

        int wait_time_;
        bool stop_flag_;

    public:
        ImageWriter(BlockingQueue<ImageData>* q_arr, size_t q_size);
        ~ImageWriter();

        bool openfiles(const vector<int>& partitions);
        void closefiles();

        void set_stop();
        void run();

    };
}

#endif