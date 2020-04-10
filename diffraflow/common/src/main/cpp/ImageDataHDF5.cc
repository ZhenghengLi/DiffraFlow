#include "ImageDataHDF5.hh"

diffraflow::ImageDataHDF5::ImageDataHDF5(): H5::CompType(sizeof(Field)) {
    //// event_time
    insertMember("event_time", HOFFSET(Field, event_time), H5::PredType::NATIVE_UINT64);
    //// alignment
    hsize_t detector_dim[] = {DET_CNT};
    H5::ArrayType alignment_t(H5::PredType::NATIVE_HBOOL, 1, detector_dim);
    insertMember("alignment", HOFFSET(Field, alignment), alignment_t);
    //// image_frame
    hsize_t image_frame_dim[] = {DET_CNT, IMAGE_H, IMAGE_W};
    H5::ArrayType image_frame_t(H5::PredType::NATIVE_FLOAT, 3, image_frame_dim);
    insertMember("image_frame", HOFFSET(Field, image_frame), image_frame_t);
    //// wait_threshold
    insertMember("wait_threshold", HOFFSET(Field, wait_threshold), H5::PredType::NATIVE_UINT64);
    //// late_arrived
    insertMember("late_arrived", HOFFSET(Field, late_arrived), H5::PredType::NATIVE_HBOOL);
}

diffraflow::ImageDataHDF5::~ImageDataHDF5() {

}