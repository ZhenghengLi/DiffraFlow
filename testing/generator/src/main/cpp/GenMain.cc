#include <iostream>
#include <H5Cpp.h>

#include "GenOptMan.hh"

using namespace std;
using namespace diffraflow;
using namespace H5;

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

    // clean
    delete[] pedestal_arr;
    delete[] gain_arr;
    delete[] threshold_arr;

    return 0;
}
