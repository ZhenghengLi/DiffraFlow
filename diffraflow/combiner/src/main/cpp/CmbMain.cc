#include <iostream>
#include <stdlib.h>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include "CmbConfig.hh"
#include "CmbImgFrmSrv.hh"
#include "CmbImgCache.hh"

using namespace std;
using namespace diffraflow;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: " << "combiner" << " <config.conf>" << endl;
        return 2;
    }

    string config_fn = argv[1];
    diffraflow::CmbConfig config;
    if (!config.load(config_fn.c_str())) {
        cout << "Failed to load CmbConfig file: " << config_fn << endl;
        return 1;
    }

    cout << endl;
    config.print();
    cout << endl;

    boost::log::core::get()->set_filter(
        boost::log::trivial::severity >= boost::log::trivial::debug
    );

    CmbImgCache image_cache;
    CmbImgFrmSrv image_frame_server(config.port, &image_cache);
    image_frame_server.serve();

    return 0;
}
