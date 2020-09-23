#ifndef __ImageFileRawW_H__
#define __ImageFileRawW_H__

#include <fstream>
#include <log4cxx/logger.h>

#include <mutex>
#include <atomic>

using std::mutex;
using std::ofstream;
using std::atomic;
using std::string;

namespace diffraflow {
    class ImageFileRawW {
    public:
        ImageFileRawW();
        ~ImageFileRawW();

        bool open(const char* filename);
        void close();
        bool write(const char* data, size_t len);
        size_t size();

    private:
        ofstream outfile_;
        atomic<size_t> image_counts_;

        mutex file_op_mtx_;
        string current_filename_;

    private:
        static log4cxx::LoggerPtr logger_;
        static const string inprogress_suffix_;
    };
} // namespace diffraflow

#endif