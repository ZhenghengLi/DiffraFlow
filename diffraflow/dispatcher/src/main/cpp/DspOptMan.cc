#include "DspOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::DspOptMan::opt_string_[] = "c:l:vh";

const option diffraflow::DspOptMan::long_opts_[] = {
    {"config", required_argument, NULL, 'c'},
    {"logconf", required_argument, NULL, 'l'},
    {"help", no_argument, NULL, 'h'},
    {"version", no_argument, NULL, 'v'},
    {NULL, no_argument, NULL, 0}
};

diffraflow::DspOptMan::DspOptMan(): OptionsManager("dispatcher") {
    config_file.clear();
    logconf_file.clear();
}

diffraflow::DspOptMan::~DspOptMan() {

}

bool diffraflow::DspOptMan::parse(int argc, char** argv) {
    while (true) {
        int opt, idx;
        opt = getopt_long(argc, argv, opt_string_, long_opts_, &idx);
        if (opt < 0) break;
        switch (opt) {
        case 'c':
            config_file = optarg;
            break;
        case 'l':
            logconf_file = optarg;
            break;
        case 'v':
            version_flag_ = true;
            return false;
        case 'h':
        case '?':
            return false;
        }
    }
    // validation check
    bool succ_flag = true;
    if (config_file.empty()) {
        cerr << "configuration file is not specified." << endl;
        succ_flag = false;
    }
    // return
    if (succ_flag) {
        return true;
    } else {
        cerr << endl;
        return false;
    }
}

void diffraflow::DspOptMan::print_help_() {
    cout << "Usage:" << endl;
    cout << "  dispatcher [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -c, --config=FILE" << setw(50) << "Configuration file" << endl;
    cout << setw(30) << "  -l, --logconf=FILE" << setw(50) << "Log configuration file" << endl;
    cout << setw(30) << "  -v, --version" << setw(50) << "print version and author information" << endl;
    cout << setw(30) << "  -h, --help" << setw(50) << "print this help" << endl;
    cout << endl;
}
