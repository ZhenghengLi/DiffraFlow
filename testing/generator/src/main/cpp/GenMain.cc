#include <iostream>
#include <fstream>
#include <cmath>
#include <H5Cpp.h>
#include <stdio.h>
#include <boost/crc.hpp>

#include "GenOptMan.hh"
#include "PrimitiveSerializer.hh"

#define FN_BUFF_SIZE 512
#define FRAME_BUFF_SISE 131096

using namespace std;
using namespace diffraflow;
using namespace H5;
using boost::crc_32_type;

DataSet get_image_dset(H5File* h5file, const string& det_path, const string& name) {
    Group det_group = h5file->openGroup(det_path);
    string det_name = det_group.getObjnameByIdx(0);
    DataSet one_dset = det_group.openDataSet(det_name + "/image/" + name);
    return one_dset;
}

int main(int argc, char** argv) {
    GenOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }

    //// read calibration data
    float* pedestal_arr = new float[3 * 512 * 128];
    float* gain_arr = new float[3 * 512 * 128];
    float* threshold_arr = new float[2 * 512 * 128];
    // open calib file
    H5File* calib_h5file = new H5File(option_man.calib_file, H5F_ACC_RDONLY);
    // read pedestal
    DataSet pedestal_dset = calib_h5file->openDataSet("pedestal");
    hsize_t pedestal_mdim[] = {1, 3, 512, 128};
    DataSpace pedestal_mspace(4, pedestal_mdim);
    DataSpace pedestal_fspace = pedestal_dset.getSpace();
    hsize_t pedestal_offset[] = {0, 0, 0, 0};
    pedestal_offset[0] = option_man.module_id;
    pedestal_fspace.selectHyperslab(H5S_SELECT_SET, pedestal_mdim, pedestal_offset);
    pedestal_dset.read(pedestal_arr, PredType::NATIVE_FLOAT, pedestal_mspace, pedestal_fspace);

    DataSet gain_dset = calib_h5file->openDataSet("gain");
    hsize_t gain_mdim[] = {1, 3, 512, 128};
    DataSpace gain_mspace(4, gain_mdim);
    DataSpace gain_fspace = gain_dset.getSpace();
    hsize_t gain_offset[] = {0, 0, 0, 0};
    gain_offset[0] = option_man.module_id;
    gain_fspace.selectHyperslab(H5S_SELECT_SET, gain_mdim, gain_offset);
    gain_dset.read(gain_arr, PredType::NATIVE_FLOAT, gain_mspace, gain_fspace);

    DataSet threshold_dset = calib_h5file->openDataSet("threshold");
    hsize_t threshold_mdim[] = {1, 2, 512, 128};
    DataSpace threshold_mspace(4, threshold_mdim);
    DataSpace threshold_fspace = threshold_dset.getSpace();
    hsize_t threshold_offset[] = {0, 0, 0, 0};
    threshold_offset[0] = option_man.module_id;
    threshold_fspace.selectHyperslab(H5S_SELECT_SET, threshold_mdim, threshold_offset);
    threshold_dset.read(threshold_arr, PredType::NATIVE_FLOAT, threshold_mspace, threshold_fspace);

    // close calib file
    calib_h5file->close();
    delete calib_h5file;
    calib_h5file = nullptr;

    //// open event num file
    H5File* event_h5file = new H5File(option_man.event_file, H5F_ACC_RDONLY);
    DataSet event_num_dset = event_h5file->openDataSet("event_num");
    int event_num = 0;
    hsize_t event_num_mdim[] = {1};
    DataSpace event_num_mspace(1, event_num_mdim);
    DataSpace event_num_fspace = event_num_dset.getSpace();
    hsize_t event_num_offset[] = {0};
    hsize_t event_num_len;
    event_num_fspace.getSimpleExtentDims(&event_num_len);

    //// open alignment file
    H5File* align_h5file = new H5File(option_man.align_file, H5F_ACC_RDONLY);
    DataSet align_idx_dset = align_h5file->openDataSet("alignment_index");
    int align_idx_arr[2];
    hsize_t align_idx_mdim[] = {1, 1, 2};
    DataSpace align_idx_mspace(3, align_idx_mdim);
    DataSpace align_idx_fspace = align_idx_dset.getSpace();
    hsize_t align_idx_offset[] = {0, 0, 0};
    align_idx_offset[1] = option_man.module_id;

    //// convert
    const string det_path = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET";
    char filename_buffer[FN_BUFF_SIZE];
    int sequential_id = -1;
    int event_counts = -1;
    int current_file_idx = -1;
    ofstream* binary_outfile = nullptr;
    H5File* data_h5file = nullptr;

    // dataset
    DataSet cellId_dset;
    uint16_t cellId = 0;
    hsize_t cellId_mdim[] = {1};
    DataSpace cellId_mspace(1, cellId_mdim);
    DataSpace cellId_fspace;
    hsize_t cellId_offset[] = {0};

    DataSet mask_dset;
    uint8_t mask_arr[512 * 128];
    hsize_t mask_mdim[] = {1, 512, 128};
    DataSpace mask_mspace(3, mask_mdim);
    DataSpace mask_fspace;
    hsize_t mask_offset[] = {0, 0, 0};

    DataSet image_dset;
    float image_arr[512 * 128];
    hsize_t image_mdim[] = {1, 512, 128};
    DataSpace image_mspace(3, image_mdim);
    DataSpace image_fspace;
    hsize_t image_offset[] = {0, 0, 0};

    char* frame_buffer = new char[FRAME_BUFF_SISE];
    crc_32_type crc_32;

    // read and convert event by event
    for (size_t event = 0; event < event_num_len; event++) {
        if (event % 100 == 0) {
            cout << "converting " << event << endl;
        }
        if (event >= 100) break;
        cout << event << endl;
        // read alignment index
        event_num_offset[0] = event;
        event_num_fspace.selectHyperslab(H5S_SELECT_SET, event_num_mdim, event_num_offset);
        event_num_dset.read(&event_num, PredType::NATIVE_INT32, event_num_mspace, event_num_fspace);
        align_idx_offset[0] = event_num;
        align_idx_fspace.selectHyperslab(H5S_SELECT_SET, align_idx_mdim, align_idx_offset);
        align_idx_dset.read(align_idx_arr, PredType::NATIVE_INT32, align_idx_mspace, align_idx_fspace);
        // cout << event_num << " (" << align_idx_arr[0] << ", " << align_idx_arr[1] << ")" << endl;
        if (align_idx_arr[0] < 0 || align_idx_arr[1] < 0) {
            cout << "unexpected alignment index: (" << align_idx_arr[0] << ", " << align_idx_arr[1] << ")" << endl;
            break;
        }
        // (re)open binary output file if needed.
        if (event_counts < 0 || event_counts >= option_man.max_events) {
            if (binary_outfile != nullptr) {
                binary_outfile->flush();
                binary_outfile->close();
                delete binary_outfile;
                binary_outfile = nullptr;
            }
            sequential_id += 1;
            snprintf(filename_buffer, FN_BUFF_SIZE, "%s/AGIPD-BIN-R0243-M%02d-S%03d.dat", option_man.output_dir.c_str(),
                option_man.module_id, sequential_id);
            binary_outfile = new ofstream(filename_buffer, ios::out | ios::binary);
            if (!binary_outfile->is_open()) {
                cerr << "failed open binary output file " << filename_buffer << endl;
                break;
            }
            event_counts = 0;
        }
        // (re)open image data file if needed.
        if (current_file_idx != align_idx_arr[0]) {
            if (data_h5file != nullptr) {
                data_h5file->close();
                delete data_h5file;
                data_h5file = nullptr;
            }
            snprintf(filename_buffer, FN_BUFF_SIZE, "%s/CORR-R0243-AGIPD%02d-S%05d.h5", option_man.data_dir.c_str(),
                option_man.module_id, align_idx_arr[0]);
            data_h5file = new H5File(filename_buffer, H5F_ACC_RDONLY);
            cellId_dset = get_image_dset(data_h5file, det_path, "cellId");
            cellId_fspace = cellId_dset.getSpace();
            mask_dset = get_image_dset(data_h5file, det_path, "mask");
            mask_fspace = mask_dset.getSpace();
            image_dset = get_image_dset(data_h5file, det_path, "data");
            image_fspace = image_dset.getSpace();
        }
        // read cellId
        cellId_offset[0] = align_idx_arr[1];
        cellId_fspace.selectHyperslab(H5S_SELECT_SET, cellId_mdim, cellId_offset);
        cellId_dset.read(&cellId, PredType::NATIVE_UINT16, cellId_mspace, cellId_fspace);
        // read mask
        mask_offset[0] = align_idx_arr[1];
        mask_fspace.selectHyperslab(H5S_SELECT_SET, mask_mdim, mask_offset);
        mask_dset.read(mask_arr, PredType::NATIVE_UINT8, mask_mspace, mask_fspace);
        // read image data
        image_offset[0] = align_idx_arr[1];
        image_fspace.selectHyperslab(H5S_SELECT_SET, image_mdim, image_offset);
        image_dset.read(image_arr, PredType::NATIVE_FLOAT, image_mspace, image_fspace);
        //// assemble frame
        gPS.serializeValue<uint32_t>(0xDEFAF127, frame_buffer + 0, 4);           // header
        gPS.serializeValue<uint16_t>(event % 65536, frame_buffer + 4, 2);        // frame index
        gPS.serializeValue<uint16_t>(option_man.module_id, frame_buffer + 6, 2); // module ID
        gPS.serializeValue<uint16_t>(cellId, frame_buffer + 8, 2);               // cell ID
        gPS.serializeValue<uint16_t>(0, frame_buffer + 10, 2);                   // status
        gPS.serializeValue<uint64_t>(event, frame_buffer + 12, 8);               // Bunch ID
        for (int row = 0; row < 512; row++) {
            for (int col = 0; col < 128; col++) {
                int idx = row * 128 + col;
                float energy = image_arr[idx] / 1000.0;
                if (mask_arr[idx] > 0 || energy < -0.1) energy = 0;
                int gain = 0;
                if (energy < threshold_arr[idx]) {
                    gain = 0;
                } else if (energy < threshold_arr[idx + 512 * 128]) {
                    gain = 1;
                } else {
                    gain = 2;
                }
                int ADC = round(energy * gain_arr[idx + gain * 512 * 128] + pedestal_arr[idx + gain * 512 * 128]);
                if (ADC < 0) ADC = 0;
                if (ADC > 16383) ADC = 16383;
                uint16_t pixel = (gain << 14) | ADC;
                gPS.serializeValue<uint16_t>(pixel, frame_buffer + 20 + idx * 2, 2);
            }
        }
        crc_32.reset();
        crc_32.process_bytes(frame_buffer + 4, 131088);
        uint32_t crc_value = crc_32.checksum();
        gPS.serializeValue<uint32_t>(crc_value, frame_buffer + 131092, 4);
        binary_outfile->write(frame_buffer, FRAME_BUFF_SISE);
    }

    // clean
    delete[] pedestal_arr;
    delete[] gain_arr;
    delete[] threshold_arr;
    delete[] frame_buffer;
    align_h5file->close();
    delete align_h5file;
    event_h5file->close();
    delete event_h5file;
    if (data_h5file != nullptr) {
        data_h5file->close();
        delete data_h5file;
    }
    if (binary_outfile != nullptr) {
        binary_outfile->flush();
        binary_outfile->close();
        delete binary_outfile;
    }

    return 0;
}
