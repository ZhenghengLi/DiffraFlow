#ifndef __ImageFileHDF5W_H__
#define __ImageFileHDF5W_H__

#include "ImageDataHDF5.hh"
#include "ImageData.hh"
#include <log4cxx/logger.h>

#include <mutex>
#include <atomic>

using std::mutex;
using std::atomic;
using std::string;

namespace diffraflow {
    class ImageFileHDF5W {
    public:
        ImageFileHDF5W(size_t buffer_size, size_t chunk_size, bool swmr = true);
        ~ImageFileHDF5W();

        bool open(const char* filename, int compress_level = -1);
        bool append(const ImageData& image_data);
        bool flush();
        void close();
        size_t size();

    private:
        bool flush_op_();

    private:
        ImageDataHDF5 image_data_hdf5_;
        ImageDataHDF5::Field* image_buffer_;

        size_t buffer_size_;
        size_t buffer_limit_;
        size_t chunk_size_;

        H5::H5File* h5file_;
        bool swmr_mode_;
        H5::DataSet imgdat_dset_;
        size_t imgdat_dset_pos_;
        hid_t imgdat_dset_id_;

        mutex file_op_mtx_;
        string current_filename_;

        atomic<size_t> image_counts_;

    private:
        static log4cxx::LoggerPtr logger_;
        static const string inprogress_suffix_;
    };
} // namespace diffraflow

#endif