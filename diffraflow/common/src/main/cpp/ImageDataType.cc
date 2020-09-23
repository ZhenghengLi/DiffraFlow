#include "ImageDataType.hh"
#include "ImageData.hh"
#include "ImageFrame.hh"
#include "Decoder.hh"

using std::cout;
using std::endl;
using std::string;

diffraflow::ImageDataType::ImageDataType() : H5::CompType(sizeof(Field)) {
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
    hsize_t frame_dim[] = {MOD_CNT, FRAME_H, FRAME_W};
    H5::ArrayType pixel_data_t(H5::PredType::NATIVE_FLOAT, 3, frame_dim);
    insertMember("pixel_data", HOFFSET(Field, pixel_data), pixel_data_t);
    //// gain_level
    H5::ArrayType gain_level_t(H5::PredType::NATIVE_UINT8, 3, frame_dim);
    insertMember("gain_level", HOFFSET(Field, gain_level), gain_level_t);
    //// late_arrived
    insertMember("late_arrived", HOFFSET(Field, late_arrived), H5::PredType::NATIVE_HBOOL);
    //// calib_level
    insertMember("calib_level", HOFFSET(Field, calib_level), H5::PredType::NATIVE_INT8);
}

diffraflow::ImageDataType::~ImageDataType() {}

bool diffraflow::ImageDataType::decode(Field& image_data, const char* buffer, const size_t len) {

    if (len < 11) return false;

    // bunch_id
    image_data.bunch_id = gDC.decode_byte<uint64_t>(buffer, 0, 7);

    // alignment
    uint16_t alignment_bits = gDC.decode_byte<uint16_t>(buffer, 8, 9);
    for (size_t i = 0; i < MOD_CNT; i++) {
        image_data.alignment[i] = (1 << (15 - i)) & alignment_bits;
    }

    // late_arrived
    image_data.late_arrived = gDC.decode_byte<uint8_t>(buffer, 10, 10);

    // frame data
    size_t current_pos = 11;
    for (size_t i = 0; i < MOD_CNT; i++) {
        if (image_data.alignment[i]) {
            if (len - current_pos < FRAME_S) return false;
            const char* frame_buffer = buffer + current_pos;
            // verification
            uint32_t header = gDC.decode_byte<uint32_t>(frame_buffer, 0, 3);
            if (header != 0xDEFAF127) {
                return false;
            }
            uint64_t bunch_id = gDC.decode_byte<uint64_t>(frame_buffer, 12, 19);
            if (bunch_id != image_data.bunch_id) {
                return false;
            }
            uint16_t module_id = gDC.decode_byte<uint16_t>(frame_buffer, 6, 7);
            if (module_id != i) {
                return false;
            }
            // cell_id
            image_data.cell_id[i] = gDC.decode_byte<uint16_t>(frame_buffer, 8, 9);
            // status
            image_data.status[i] = gDC.decode_byte<uint16_t>(frame_buffer, 10, 11);

            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    size_t offset = 20 + 2 * (h * FRAME_W + w);
                    // pixel_data
                    image_data.gain_level[i][h][w] = gDC.decode_bit<uint8_t>(frame_buffer + offset, 0, 1);
                    // gain_level
                    image_data.pixel_data[i][h][w] = gDC.decode_bit<uint16_t>(frame_buffer + offset, 2, 15);
                }
            }

            current_pos += FRAME_S;
        } else {
            // cell_id
            image_data.cell_id[i] = 0;
            // status
            image_data.status[i] = 0;

            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    // pixel_data
                    image_data.pixel_data[i][h][w] = 0;
                    // gain_level
                    image_data.gain_level[i][h][w] = 0;
                }
            }
        }
    }

    // calib_level
    image_data.calib_level = 0;

    return true;
}

void diffraflow::ImageDataType::print(const Field& image_data, ostream& out) {
    out << "bunch_id: " << image_data.bunch_id << endl;
    out << "late_arrived: " << image_data.late_arrived << endl;
    out << "alignment: [";
    for (size_t i = 0; i < MOD_CNT; i++) {
        if (i > 0) out << ", ";
        out << image_data.alignment[i];
    }
    out << "]" << endl;
}

void diffraflow::ImageDataType::convert(const Field& image_data_arr, ImageData& image_data_obj) {
    image_data_obj.bunch_id = image_data_arr.bunch_id;
    image_data_obj.late_arrived = image_data_arr.late_arrived;
    image_data_obj.calib_level = image_data_arr.calib_level;
    image_data_obj.alignment_vec.resize(MOD_CNT);
    image_data_obj.image_frame_vec.resize(MOD_CNT);
    for (size_t i = 0; i < MOD_CNT; i++) {
        image_data_obj.alignment_vec[i] = image_data_arr.alignment[i];
        if (image_data_arr.alignment[i]) {
            image_data_obj.image_frame_vec[i] = make_shared<ImageFrame>();
            image_data_obj.image_frame_vec[i]->bunch_id = image_data_arr.bunch_id;
            image_data_obj.image_frame_vec[i]->module_id = i;
            image_data_obj.image_frame_vec[i]->cell_id = image_data_arr.cell_id[i];
            image_data_obj.image_frame_vec[i]->status = image_data_arr.status[i];
            image_data_obj.image_frame_vec[i]->pixel_data.resize(FRAME_L);
            image_data_obj.image_frame_vec[i]->gain_level.resize(FRAME_L);
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    size_t pos = h * FRAME_W + w;
                    image_data_obj.image_frame_vec[i]->pixel_data[pos] = image_data_arr.pixel_data[i][h][w];
                    image_data_obj.image_frame_vec[i]->gain_level[pos] = image_data_arr.gain_level[i][h][w];
                }
            }
        } else {
            image_data_obj.image_frame_vec[i] = nullptr;
        }
    }
}
