#include "ImageFileHDF5W.hh"
#include "ctime"
#include <boost/algorithm/string.hpp>
#include <string>

using std::string;

log4cxx::LoggerPtr diffraflow::ImageFileHDF5W::logger_
    = log4cxx::Logger::getLogger("ImageFileHDF5W");

diffraflow::ImageFileHDF5W::ImageFileHDF5W(size_t buffer_size, size_t chunk_size) {
    buffer_size_ = (buffer_size > 10 ? buffer_size : 10);
    chunk_size_ = (chunk_size >= buffer_size ? chunk_size : buffer_size);

    image_buffer_ = new ImageDataHDF5::Field[buffer_size_];
    buffer_limit_ = 0;

    h5file_ = nullptr;

}

diffraflow::ImageFileHDF5W::~ImageFileHDF5W() {
    delete [] image_buffer_;
    image_buffer_ = nullptr;
}

bool diffraflow::ImageFileHDF5W::open(const char* filename, int compress_level) {
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

    try {
        h5file_ = new H5::H5File(filename, H5F_ACC_EXCL | H5F_ACC_SWMR_WRITE);
        // add create time attribute to the file
        H5::Attribute file_create_time = h5file_->createAttribute("create_time", string_type, string_attr_space);
        time_t now_time = time(NULL);
        string now_time_string = boost::trim_copy(string(ctime(&now_time)));
        file_create_time.write(string_type, H5std_string(now_time_string));
        // create dataset for image data
        imgdat_dset_ = h5file_->createDataSet("image_data", image_data_hdf5, imgdat_file_space, imgdat_dset_crtpars);
        // close and reopen
        h5file_->close();
        delete h5file_;
        h5file_  = new H5::H5File(filename, H5F_ACC_RDWR | H5F_ACC_SWMR_WRITE);
        imgdat_dset_ = h5file_->openDataSet("image_data");
    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when opening HDF5 file: " << filename);
        e.printError();
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when opening HDF5 file: " << filename);
        return false;
    }

    return true;
}

bool diffraflow::ImageFileHDF5W::append(const ImageData& image_data) {

    return true;
}

void diffraflow::ImageFileHDF5W::flush() {

}

void diffraflow::ImageFileHDF5W::close() {

}