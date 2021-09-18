#include "FtrOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::FtrOptMan::opt_string_[] = "i:o:c:l:vh";

// clang-format off
const option diffraflow::FtrOptMan::long_opts_[] = {
    {"datfile", required_argument, NULL, 'i'},
    {"outfile", required_argument, NULL, 'o'},
    {"config",  required_argument, NULL, 'c'},
    {"logconf", required_argument, NULL, 'l'},
    {"help",    no_argument,       NULL, 'h'},
    {"version", no_argument,       NULL, 'v'},
    {NULL,      no_argument,       NULL, 0}
};
// clang-format on

diffraflow::FtrOptMan::FtrOptMan() : OptionsManager("generator") {
    data_file.clear();
    output_file.clear();
    config_file.clear();
}

diffraflow::FtrOptMan::~FtrOptMan() {}

bool diffraflow::FtrOptMan::parse(int argc, char** argv) {
    while (true) {
        int opt, idx;
        opt = getopt_long(argc, argv, opt_string_, long_opts_, &idx);
        if (opt < 0) break;
        switch (opt) {
        case 'i':
            data_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
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
    if (data_file.empty()) {
        cerr << "data file is not set." << endl;
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

void diffraflow::FtrOptMan::print_help_() {
    // clang-format off
    cout << "Usage:" << endl;
    cout << "  " << software_name_ << " [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -i, --datfile=FILE"  << setw(50) << "data file" << endl;
    cout << setw(30) << "  -o, --outfile=FILE"  << setw(50) << "output file" << endl;
    cout << setw(30) << "  -c, --config=FILE"   << setw(50) << "config file" << endl;
    cout << setw(30) << "  -l, --logconf=FILE"  << setw(50) << "log configuration file" << endl;
    cout << setw(30) << "  -v, --version"       << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help"          << setw(50) << "print this help" << endl;
    cout << endl;
    // clang-format on
}
