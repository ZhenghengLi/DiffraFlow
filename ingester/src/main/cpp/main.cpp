#include <iostream>
#include <stdlib.h>

#include "Configuration.hpp"
#include "ImageFrame.hpp"

#include "BlockingQueue.hpp"

using namespace std;
using namespace shine;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: " << "ingester" << " <config.conf>" << endl;
        return 2;
    }

    string config_fn = argv[1];
    shine::Configuration config;
    if (!config.load(config_fn.c_str())) {
        cout << "Failed to load configuration file: " << config_fn << endl;
        return 1;
    }

    cout << endl;
    config.print();
    cout << endl;

    return 0;
}
