#include "IngOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::IngOptMan::opt_string_[] = "c:l:vh";

const option diffraflow::IngOptMan::long_opts_[] = {
    {"config", required_argument, NULL, 'c'},
    {"logconf", required_argument, NULL, 'l'},
    {"help", no_argument, NULL, 'h'},
    {"version", no_argument, NULL, 'v'},
    {NULL, no_argument, NULL, 0}
};

diffraflow::IngOptMan::IngOptMan(): OptionsManager("ingester") {
    config_file.clear();
    logconf_file.clear();
}

diffraflow::IngOptMan::~IngOptMan() {

}

bool diffraflow::IngOptMan::parse(int argc, char** argv) {
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
        cerr << "configuration file should be set." << endl;
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

void diffraflow::IngOptMan::print_help_() {
    cout << "Usage:" << endl;
    cout << "  " << software_name_ << " [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -c, --config=FILE" << setw(50) << "configuration file" << endl;
    cout << setw(30) << "  -l, --logconf=FILE" << setw(50) << "log configuration file" << endl;
    cout << setw(30) << "  -v, --version" << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help" << setw(50) << "print this help" << endl;
    cout << endl;
}
