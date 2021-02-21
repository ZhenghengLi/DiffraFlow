#include "ImageDataHDF5.hh"
#include <iostream>

using std::cout;
using std::endl;

diffraflow::ImageDataHDF5::ImageDataHDF5() : H5::CompType(sizeof(Field)) {
    //// bunch_id
    insertMember("bunch_id", HOFFSET(Field, bunch_id), H5::PredType::NATIVE_UINT64);
    //// alignment
    hsize_t module_dim[] = {MOD_CNT};
    H5::ArrayType alignment_t(H5::PredType::NATIVE_HBOOL, 1, module_dim);
    insertMember("alignment", HOFFSET(Field, alignment), alignment_t);
    //// cell_id
    H5::ArrayType cell_id_t(H5::PredType::NATIVE_INT16, 1, module_dim);
    insertMember("cell_id", HOFFSET(Field, cell_id), cell_id_t);
    //// status
    H5::ArrayType status_t(H5::PredType::NATIVE_UINT16, 1, module_dim);
    insertMember("status", HOFFSET(Field, status), status_t);
    //// image_frame
    hsize_t frame_dim[] = {MOD_CNT, IMAGE_H, IMAGE_W};
    H5::ArrayType pixel_data_t(H5::PredType::NATIVE_FLOAT, 3, frame_dim);
    insertMember("pixel_data", HOFFSET(Field, pixel_data), pixel_data_t);
    //// gain_level
    H5::ArrayType gain_level_t(H5::PredType::NATIVE_UINT8, 3, frame_dim);
    insertMember("gain_level", HOFFSET(Field, gain_level), gain_level_t);
    //// late_arrived
    insertMember("late_arrived", HOFFSET(Field, late_arrived), H5::PredType::NATIVE_HBOOL);
}

diffraflow::ImageDataHDF5::~ImageDataHDF5() {}

void diffraflow::ImageDataHDF5::convert_image(const ImageData& imgdat_obj, Field& imgdat_st) {

    // bunch_id
    imgdat_st.bunch_id = imgdat_obj.bunch_id;

    for (size_t i = 0; i < MOD_CNT; i++) {

        // alignment, cell_id, status
        if (i < imgdat_obj.alignment_vec.size()) {
            imgdat_st.alignment[i] = imgdat_obj.alignment_vec[i];
            if (imgdat_obj.alignment_vec[i]) {
                imgdat_st.cell_id[i] = imgdat_obj.image_frame_vec[i]->cell_id;
                imgdat_st.status[i] = imgdat_obj.image_frame_vec[i]->status;
            } else {
                imgdat_st.cell_id[i] = -1;
                imgdat_st.status[i] = 0;
            }
        } else {
            imgdat_st.alignment[i] = 0;
            imgdat_st.cell_id[i] = -1;
            imgdat_st.status[i] = 0;
        }

        // pixel_data, gain_level
        for (size_t h = 0; h < IMAGE_H; h++) {
            for (size_t w = 0; w < IMAGE_W; w++) {
                size_t pos = h * IMAGE_W + w;
                if (i < imgdat_obj.alignment_vec.size() && imgdat_obj.alignment_vec[i] && pos < 65536) {
                    imgdat_st.pixel_data[i][h][w] = imgdat_obj.image_frame_vec[i]->pixel_data[pos];
                    imgdat_st.gain_level[i][h][w] = imgdat_obj.image_frame_vec[i]->gain_level[pos];
                } else {
                    imgdat_st.pixel_data[i][h][w] = 0;
                    imgdat_st.gain_level[i][h][w] = 0;
                }
            }
        }
    }

    // late_arrived
    imgdat_st.late_arrived = imgdat_obj.late_arrived;
}