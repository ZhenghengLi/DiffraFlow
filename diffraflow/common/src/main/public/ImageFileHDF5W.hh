#ifndef __ImageFileHDF5W_H__
#define __ImageFileHDF5W_H__

#include "ImageDataHDF5.hh"
#include "ImageData.hh"
#include <log4cxx/logger.h>

#define DET_CNT 4
#define IMAGE_H 6
#define IMAGE_W 8

namespace diffraflow {
    class ImageFileHDF5W {
    public:
        ImageFileHDF5W(size_t buffer_size, size_t chunk_size);
        ~ImageFileHDF5W();

        bool open(const char* filename, int compress_level = -1);
        bool append(const ImageData& image_data);
        void flush();
        void close();

    public:
        ImageDataHDF5   image_data_hdf5;

    private:
        ImageDataHDF5::Field*   image_buffer_;
        size_t                  buffer_limit_;

        H5::H5File*             h5file_;
        H5::DataSet             imgdat_dset_;
        size_t                  imgdat_dset_pos_;

    private:
        size_t buffer_size_;
        size_t chunk_size_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif