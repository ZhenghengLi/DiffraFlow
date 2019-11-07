#include <iostream>
#include <stdlib.h>

#include "Configuration.hh"
#include "ImageFrameServer.hh"
#include "ImageCache.hh"

using namespace std;
using namespace shine;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: " << "combiner" << " <config.conf>" << endl;
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

    ImageCache image_cache;
    ImageFrameServer image_frame_server(&image_cache);
    image_frame_server.serve(config.port);

    return 0;
}
