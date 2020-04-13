#ifndef __ImageFileHDF5W_H__
#define __ImageFileHDF5W_H__

#include "ImageDataHDF5.hh"
#include "ImageData.hh"
#include <log4cxx/logger.h>

#include <mutex>

#define DET_CNT 4
#define IMAGE_H 6
#define IMAGE_W 8

using std::mutex;

namespace diffraflow {
    class ImageFileHDF5W {
    public:
        ImageFileHDF5W(size_t buffer_size, size_t chunk_size);
        ~ImageFileHDF5W();

        bool open(const char* filename, int compress_level = -1);
        bool append(const ImageData& image_data);
        bool flush();
        void close();

    public:
        ImageDataHDF5   image_data_hdf5;

    private:
        ImageDataHDF5::Field*   image_buffer_;

        H5::H5File*             h5file_;
        H5::DataSet             imgdat_dset_;
        size_t                  imgdat_dset_pos_;
        hid_t                   imgdat_dset_id_;

        mutex                   file_op_mtx_;

    private:
        size_t buffer_size_;
        size_t buffer_pos_;
        size_t chunk_size_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif