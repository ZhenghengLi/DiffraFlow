#include "ImageFileHDF5.hh"

diffraflow::ImageFileHDF5::ImageFileHDF5(
    size_t chunk_size, size_t batch_size, int compress_level) {
    chunk_size_ = chunk_size;
    batch_size_ = batch_size;
    compress_level_ = compress_level;

}

diffraflow::ImageFileHDF5::~ImageFileHDF5() {

}