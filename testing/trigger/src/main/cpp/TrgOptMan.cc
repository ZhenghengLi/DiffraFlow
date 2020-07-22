#include "TrgOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::TrgOptMan::opt_string_[] = "s:vh";

// clang-format off
const option diffraflow::TrgOptMan::long_opts_[] = {
    {"senderlist", required_argument, NULL, 's'},
    {"help", no_argument, NULL, 'h'},
    {"version", no_argument, NULL, 'v'},
    {NULL, no_argument, NULL, 0}
};
// clang-format on

diffraflow::TrgOptMan::TrgOptMan() : OptionsManager("trigger") {
    // init
    sender_list_file.clear();
}

diffraflow::TrgOptMan::~TrgOptMan() {}

bool diffraflow::TrgOptMan::parse(int argc, char** argv) {
    while (true) {
        int opt, idx;
        opt = getopt_long(argc, argv, opt_string_, long_opts_, &idx);
        if (opt < 0) break;
        switch (opt) {
        case 's':
            sender_list_file = optarg;
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
    if (sender_list_file.empty()) {
        cerr << "sender list file should be set." << endl;
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

void diffraflow::TrgOptMan::print_help_() {
    // clang-format off
    cout << "Usage:" << endl;
    cout << "  " << software_name_ << " [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -s, --senderlist=FILE"  << setw(50) << "sender list file" << endl;
    cout << setw(30) << "  -v, --version"          << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help"             << setw(50) << "print this help" << endl;
    cout << endl;
    // clang-format on
}
