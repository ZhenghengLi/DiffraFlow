#ifndef __ImageFileHDF5W_H__
#define __ImageFileHDF5W_H__

#include "ImageDataType.hh"
#include <log4cxx/logger.h>

#include <mutex>
#include <atomic>

using std::mutex;
using std::atomic;
using std::string;

namespace diffraflow {
    class ImageFileHDF5W {
    public:
        ImageFileHDF5W(size_t chunk_size, bool swmr = true);
        ~ImageFileHDF5W();

        bool open(const char* filename, int compress_level = -1);
        bool write(const ImageDataField& image_data);
        bool flush();
        void close();
        size_t size();

    private:
        ImageDataType image_data_type_;

        H5::H5File* h5file_;
        size_t chunk_size_;
        bool swmr_mode_;
        H5::DataSet imgdat_dset_;
        size_t imgdat_dset_pos_;
        hid_t imgdat_dset_id_;

        mutex file_op_mtx_;
        string current_filename_;
        string inprogress_suffix_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif