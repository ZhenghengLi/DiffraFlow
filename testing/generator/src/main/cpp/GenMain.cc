#include <iostream>
#include <fstream>
#include <H5Cpp.h>

#include "GenOptMan.hh"

using namespace std;
using namespace diffraflow;
using namespace H5;

DataSet get_image_dset(H5File* h5file, const string& det_path, const string& name) {
    Group det_group = h5file->openGroup(det_path);
    string det_name = det_group.getObjnameByIdx(0);
    cout << "det_name: " << det_name << endl;
    DataSet one_dset = det_group.openDataSet(det_name + "/image/" + name);
    cout << "one_dset: " << one_dset.getObjName() << endl;
    return one_dset;
}

int main(int argc, char** argv) {
    GenOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }

    // cout << "data_dir: " << option_man.data_dir << endl;
    // cout << "module_id: " << option_man.module_id << endl;
    // cout << "output_dir: " << option_man.output_dir << endl;
    // cout << "calib_file: " << option_man.calib_file << endl;
    // cout << "align_file: " << option_man.align_file << endl;
    // cout << "event_file: " << option_man.event_file << endl;
    // cout << "max_events: " << option_man.max_events << endl;

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

    //// open alignment file
    H5File* align_h5file = new H5File(option_man.align_file, H5F_ACC_RDONLY);
    DataSet align_idx_dset = align_h5file->openDataSet("alignment_index");
    int align_idx_arr[16 * 2];
    hsize_t align_idx_mdim[] = {1, 16, 2};
    DataSpace align_idx_mspace(3, align_idx_mdim);
    DataSpace align_idx_fspace = align_idx_dset.getSpace();
    hsize_t align_idx_offset[] = {0, 0, 0};

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

    //// convert
    string det_path = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET";
    int sequential_id = -1;
    int event_counts = -1;
    ofstream* binary_outfile = nullptr;
    // H5File* data_h5file = new H5File(option_man.data_dir, H5F_ACC_RDONLY);
    // DataSet image_data_dset = get_image_dset(data_h5file, det_path, "data");
    for (size_t event = 0; event < event_num_len; event++) {
        if (event % 100 == 0) {
            cout << "converting " << event << endl;
        }
        event_num_offset[0] = event;
        event_num_fspace.selectHyperslab(H5S_SELECT_SET, event_num_mdim, event_num_offset);
        event_num_dset.read(&event_num, PredType::NATIVE_INT32, event_num_mspace, event_num_fspace);
        cout << event_num << endl;
    }

    // clean
    delete[] pedestal_arr;
    delete[] gain_arr;
    delete[] threshold_arr;

    return 0;
}
