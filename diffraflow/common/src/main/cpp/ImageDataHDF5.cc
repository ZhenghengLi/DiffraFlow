#include "ImageDataHDF5.hh"

diffraflow::ImageDataHDF5::ImageDataHDF5() : H5::CompType(sizeof(Field)) {
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

diffraflow::ImageDataHDF5::~ImageDataHDF5() {}

void diffraflow::ImageDataHDF5::convert_image(const ImageData& imgdat_obj, Field& imgdat_st) {
    // event_time
    imgdat_st.event_time = imgdat_obj.event_time;
    for (size_t i = 0; i < DET_CNT; i++) {
        // alighment
        if (i < imgdat_obj.alignment_vec.size()) {
            imgdat_st.alignment[i] = imgdat_obj.alignment_vec[i];
        } else {
            imgdat_st.alignment[i] = 0;
        }
        // image frame
        if (i < imgdat_obj.image_frame_vec.size()) {
            for (size_t h = 0; h < IMAGE_H; h++) {
                if (h < imgdat_obj.image_frame_vec[i].image_height) {
                    for (size_t w = 0; w < IMAGE_W; w++) {
                        size_t idx = h * imgdat_obj.image_frame_vec[i].image_width + w;
                        if (w < imgdat_obj.image_frame_vec[i].image_width) {
                            imgdat_st.image_frame[i][h][w] = imgdat_obj.image_frame_vec[i].image_frame[idx];
                        } else {
                            imgdat_st.image_frame[i][h][w] = 0;
                        }
                    }
                } else {
                    for (size_t w = 0; w < IMAGE_W; w++) {
                        imgdat_st.image_frame[i][h][w] = 0;
                    }
                }
            }
        } else {
            for (size_t h = 0; h < IMAGE_H; h++) {
                for (size_t w = 0; w < IMAGE_W; w++) {
                    imgdat_st.image_frame[i][h][w] = 0;
                }
            }
        }
    }
    // wait_threshold
    imgdat_st.wait_threshold = imgdat_obj.wait_threshold;
    // late_arrived
    imgdat_st.late_arrived = imgdat_obj.late_arrived;
}