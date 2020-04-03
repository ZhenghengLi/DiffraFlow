#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>
#include <stdexcept>

using std::cout;
using std::endl;
using std::cerr;
using std::copy;

diffraflow::ImageData::ImageData(uint32_t numOfDet) {
    event_time = 0;
    status_vec.resize(numOfDet, false);
    imgfrm_vec.resize(numOfDet);
    late_arrived = false;
    wait_threshold = 0;
}

diffraflow::ImageData::~ImageData() {

}

bool diffraflow::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= imgfrm_vec.size()) return false;
    status_vec[index] = true;
    imgfrm_vec[index] = imgfrm;
    return true;
}

void diffraflow::ImageData::print() {
    for (size_t i = 0; i < imgfrm_vec.size(); i++) {
        cout << "ImageData[" << i << "]:" << endl;
        imgfrm_vec[i].print();
    }
    cout << "late_arrived: " << late_arrived << endl;
}
