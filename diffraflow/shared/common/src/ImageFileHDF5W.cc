#include "ImageFileHDF5W.hh"
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <string>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;
namespace bs = boost::system;

using std::string;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::ImageFileHDF5W::logger_ = log4cxx::Logger::getLogger("ImageFileHDF5W");

diffraflow::ImageFileHDF5W::ImageFileHDF5W(size_t chunk_size, bool swmr) {
    h5file_ = nullptr;
    chunk_size_ = (chunk_size > 1 ? chunk_size : 1);
    swmr_mode_ = swmr;
    if (swmr_mode_) {
        inprogress_suffix_ = ".inprogress_swmr";
    } else {
        inprogress_suffix_ = ".inprogress";
    }
}

diffraflow::ImageFileHDF5W::~ImageFileHDF5W() { close(); }

bool diffraflow::ImageFileHDF5W::open(const char* filename, int compress_level) {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ != nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is already opened.");
        return false;
    }
    H5::Exception::dontPrint();

    hsize_t imgdat_file_dim[] = {0};
    hsize_t imgdat_file_dim_max[] = {H5S_UNLIMITED};
    H5::DataSpace imgdat_file_space(1, imgdat_file_dim, imgdat_file_dim_max);

    H5::DSetCreatPropList imgdat_dset_crtpars;
    hsize_t imgdat_dset_chunk_dim[1];
    imgdat_dset_chunk_dim[0] = chunk_size_;
    imgdat_dset_crtpars.setChunk(1, imgdat_dset_chunk_dim);
    if (compress_level > 0 && compress_level <= 9) {
        imgdat_dset_crtpars.setDeflate(compress_level);
    }

    hsize_t string_attr_dim[] = {1};
    H5::DataSpace string_attr_space(1, string_attr_dim);
    H5::StrType string_type(0, H5T_VARIABLE);

    string filename_inprogress = filename + inprogress_suffix_;

    try {
        if (swmr_mode_) {
            h5file_ = new H5::H5File(filename_inprogress, H5F_ACC_EXCL | H5F_ACC_SWMR_WRITE);
        } else {
            h5file_ = new H5::H5File(filename_inprogress, H5F_ACC_EXCL);
        }
        current_filename_ = filename;
        // add create time attribute to the file
        H5::Attribute file_create_time = h5file_->createAttribute("create_time", string_type, string_attr_space);
        time_t now_time = time(NULL);
        string now_time_string = boost::trim_copy(string(ctime(&now_time)));
        file_create_time.write(string_type, H5std_string(now_time_string));
        // create dataset for image data
        imgdat_dset_ = h5file_->createDataSet("image_data", image_data_type_, imgdat_file_space, imgdat_dset_crtpars);
        imgdat_dset_id_ = imgdat_dset_.getId();
        imgdat_dset_pos_ = 0;
        // swmr needs close and reopen
        if (swmr_mode_) {
            h5file_->close();
            delete h5file_;
            h5file_ = new H5::H5File(filename_inprogress, H5F_ACC_RDWR | H5F_ACC_SWMR_WRITE);
            imgdat_dset_ = h5file_->openDataSet("image_data");
            imgdat_dset_id_ = imgdat_dset_.getId();
        }
    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(
            logger_, "found error when opening HDF5 file " << filename_inprogress << " : " << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when opening HDF5 file: " << filename_inprogress);
        return false;
    }

    return true;
}

bool diffraflow::ImageFileHDF5W::flush() {
    lock_guard<mutex> lg(file_op_mtx_);
    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return false;
    }
    h5file_->flush(H5F_SCOPE_LOCAL);
    return true;
}

bool diffraflow::ImageFileHDF5W::write(const ImageDataField& image_data) {
    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ == nullptr) {
        LOG4CXX_ERROR(logger_, "hdf5 file is not opened.");
        return false;
    }

    try {
        // extend file space for one image
        hsize_t imgdat_dset_size[1];
        imgdat_dset_size[0] = imgdat_dset_pos_ + 1;
        imgdat_dset_.extend(imgdat_dset_size);
        // write the image data
        hsize_t file_offset[1];
        file_offset[0] = imgdat_dset_pos_;
        hsize_t memory_dim[1] = {1};
        H5::DataSpace file_space = imgdat_dset_.getSpace();
        file_space.selectHyperslab(H5S_SELECT_SET, memory_dim, file_offset);
        H5::DataSpace memory_space(1, memory_dim);
        imgdat_dset_.write(&image_data, image_data_type_, memory_space, file_space);
        // SWMR flush
        if (swmr_mode_) {
            H5Dflush(imgdat_dset_id_);
        }
        // advance position
        imgdat_dset_pos_++;
        LOG4CXX_DEBUG(logger_, "done a flush.");
        return true;
    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when flushing data : " << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when flushing data.");
        return false;
    }
}

void diffraflow::ImageFileHDF5W::close() {

    lock_guard<mutex> lg(file_op_mtx_);

    if (h5file_ != nullptr) {
        h5file_->flush(H5F_SCOPE_LOCAL);
        h5file_->close();
        delete h5file_;
        h5file_ = nullptr;
    }

    bf::path file_current_path(current_filename_ + inprogress_suffix_);
    bf::path file_target_path(current_filename_);
    bs::error_code ec;
    if (bf::exists(file_current_path)) {
        bf::rename(file_current_path, file_target_path, ec);
        if (ec != bs::errc::success) {
            LOG4CXX_WARN(
                logger_, "failed to rename file " << file_current_path.string() << " with error: " << ec.message());
        }
    }
}

size_t diffraflow::ImageFileHDF5W::size() {
    lock_guard<mutex> lg(file_op_mtx_);
    return imgdat_dset_pos_;
}