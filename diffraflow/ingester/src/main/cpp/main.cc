#include <iostream>
#include "ImageData.hh"
#include "ImageWriter.hh"

using namespace std;
using namespace shine;

int main(int argc, char** argv) {
    cout << "I am ingester." << endl;

    ImageData img_data;
    img_data.serialize(nullptr, 0);

    return 0;
}