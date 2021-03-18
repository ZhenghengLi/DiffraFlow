#include "Calibration.hh"

void diffraflow::Calibration::do_calib_cpu(ImageDataField* image_data_host, CalibDataField* calib_data_host) {

    if (image_data_host == nullptr) return;
    if (calib_data_host == nullptr) return;
    if (image_data_host->calib_level == 1) return;

    for (size_t m = 0; m < MOD_CNT; m++) {
        if (image_data_host->alignment[m]) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    size_t l = image_data_host->gain_level[m][h][w];
                    if (l < GLV_CNT) {
                        image_data_host->pixel_data[m][h][w] =
                            (image_data_host->pixel_data[m][h][w] - calib_data_host->pedestal[m][l][h][w]) *
                            calib_data_host->gain[m][l][h][w];
                    }
                }
            }
        }
    }

    image_data_host->calib_level = 1;
}