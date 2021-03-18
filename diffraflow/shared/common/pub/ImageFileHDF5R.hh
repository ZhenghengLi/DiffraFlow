#ifndef __ImageFileHDF5R_H__
#define __ImageFileHDF5R_H__

#include "ImageDataType.hh"
#include "ImageData.hh"
#include <log4cxx/logger.h>

#include <mutex>

using std::mutex;
using std::string;

namespace diffraflow {
    class ImageFileHDF5R {
    public:
        ImageFileHDF5R(size_t buffer_size, bool swmr = true);
        ~ImageFileHDF5R();

        bool open(const char* filename);
        void close();
        bool next_batch();
        bool next_image(ImageDataField& imgdat_st);
        size_t image_dset_size();
        string create_time();

    private:
        ImageDataType image_data_type_;
        ImageDataField* image_buffer_;

        size_t buffer_size_;
        size_t buffer_limit_;
        size_t buffer_pos_;

        H5::H5File* h5file_;
        bool swmr_mode_;
        string file_create_time_;
        H5::DataSet imgdat_dset_;
        hid_t imgdat_dset_id_;
        size_t imgdat_dset_pos_;
        size_t imgdat_dset_size_;

        mutex file_op_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif