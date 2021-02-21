#include "ImageFileHDF5R.hh"
#include "ctime"
#include <boost/algorithm/string.hpp>
#include <string>

using std::string;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::ImageFileHDF5R::logger_ = log4cxx::Logger::getLogger("ImageFileHDF5R");

diffraflow::ImageFileHDF5R::ImageFileHDF5R(size_t buffer_size, bool swmr) {
    buffer_size_ = (buffer_size > 10 ? buffer_size : 10);

    image_buffer_ = new ImageDataHDF5::Field[buffer_size_];
    buffer_limit_ = 0;
    buffer_pos_ = 0;

    h5file_ = nullptr;
    swmr_mode_ = swmr;
}

diffraflow::ImageFileHDF5R::~ImageFileHDF5R() {
    delete[] image_buffer_;
    image_buffer_ = nullptr;
}

bool diffraflow::ImageFileHDF5R::open(const char* filename) {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ != nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is alread opened.");
        return false;
    }
    H5::Exception::dontPrint();

    hsize_t string_attr_dim[] = {1};
    H5::DataSpace string_attr_space(1, string_attr_dim);
    H5::StrType string_type(0, H5T_VARIABLE);

    try {
        if (swmr_mode_) {
            h5file_ = new H5::H5File(filename, H5F_ACC_RDONLY | H5F_ACC_SWMR_READ);
        } else {
            h5file_ = new H5::H5File(filename, H5F_ACC_RDONLY);
        }
        // read file create time
        H5::Attribute file_create_time_attr = h5file_->openAttribute("create_time");
        file_create_time_attr.read(string_type, file_create_time_);
        // create dataset for image data
        imgdat_dset_ = h5file_->openDataSet("image_data");
        imgdat_dset_id_ = imgdat_dset_.getId();
        imgdat_dset_pos_ = 0;
        buffer_pos_ = 0;
        buffer_limit_ = 0;
        // read dataset size
        H5::DataSpace file_space = imgdat_dset_.getSpace();
        hsize_t current_dim[1] = {0};
        file_space.getSimpleExtentDims(current_dim);
        imgdat_dset_size_ = current_dim[0];
    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when opening HDF5 file " << filename << " : " << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when opening HDF5 file: " << filename);
        return false;
    }

    return true;
}

void diffraflow::ImageFileHDF5R::close() {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ != nullptr) {
        h5file_->close();
        delete h5file_;
        h5file_ = nullptr;
    }
}

bool diffraflow::ImageFileHDF5R::next_batch() {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return false;
    }

    try {
        if (swmr_mode_) {
            H5Drefresh(imgdat_dset_id_);
        }
        H5::DataSpace file_space = imgdat_dset_.getSpace();
        hsize_t current_dim[1] = {0};
        file_space.getSimpleExtentDims(current_dim);
        imgdat_dset_size_ = current_dim[0];
        int remained_size = imgdat_dset_size_ - imgdat_dset_pos_;
        if (remained_size <= 0) {
            return false;
        }
        buffer_limit_ = (buffer_size_ <= remained_size ? buffer_size_ : remained_size);
        hsize_t file_offset[1];
        file_offset[0] = imgdat_dset_pos_;
        hsize_t memory_dim[1];
        memory_dim[0] = buffer_limit_;
        file_space.selectHyperslab(H5S_SELECT_SET, memory_dim, file_offset);
        H5::DataSpace memory_space(1, memory_dim);
        imgdat_dset_.read(image_buffer_, image_data_hdf5_, memory_space, file_space);
        buffer_pos_ = 0;
        imgdat_dset_pos_ += buffer_limit_;
        return true;
    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when reading data : " << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when reading data.");
        return false;
    }
}

bool diffraflow::ImageFileHDF5R::next_image(ImageDataHDF5::Field& imgdat_st) {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return false;
    }

    if (buffer_pos_ < buffer_limit_) {
        imgdat_st = image_buffer_[buffer_pos_];
        buffer_pos_++;
        return true;
    } else {
        return false;
    }
}

size_t diffraflow::ImageFileHDF5R::image_dset_size() {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return false;
    }

    return imgdat_dset_size_;
}

string diffraflow::ImageFileHDF5R::create_time() {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return "";
    }

    return file_create_time_;
}
