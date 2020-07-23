#include "TrgOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::TrgOptMan::opt_string_[] = "s:l:e:c:i:d:vh";

// clang-format off
const option diffraflow::TrgOptMan::long_opts_[] = {
    {"senderlist", required_argument, NULL, 's'},
    {"logconf", required_argument, NULL, 'l'},
    {"startevent", required_argument, NULL, 'e'},
    {"eventcount", required_argument, NULL, 'c'},
    {"interval", required_argument, NULL, 'i'},
    {"senderid", required_argument, NULL, 'd'},
    {"help", no_argument, NULL, 'h'},
    {"version", no_argument, NULL, 'v'},
    {NULL, no_argument, NULL, 0}
};
// clang-format on

diffraflow::TrgOptMan::TrgOptMan() : OptionsManager("trigger") {
    // init
    sender_list_file.clear();
    logconf_file.clear();
    start_event_index = -1;
    event_count = -1;
    interval_microseconds = -1;
    sender_id = 0;
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
        case 'l':
            logconf_file = optarg;
            break;
        case 'e':
            start_event_index = atoi(optarg);
            break;
        case 'c':
            event_count = atoi(optarg);
            break;
        case 'i':
            interval_microseconds = atoi(optarg);
            break;
        case 'd':
            sender_id = atoi(optarg);
            break;
        case 'v':
            version_flag_ = true;
            return false;
        case 'h':
        case '?':
            return false;
        }
    }
    // correction
    if (sender_id < 0) {
        cout << "sender_id < 0, use 0 instead." << endl;
        sender_id = 0;
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
    cout << setw(30) << "  -l, --logconf=FILE"     << setw(50) << "log configuration file" << endl;
    cout << setw(30) << "  -e, --startevent=UINT"  << setw(50) << "start event index" << endl;
    cout << setw(30) << "  -c, --eventcount=UINT"  << setw(50) << "event count" << endl;
    cout << setw(30) << "  -i, --interval=UINT"    << setw(50) << "interval microseconds" << endl;
    cout << setw(30) << "  -d, --senderid=UINT"    << setw(50) << "sender id, default is 0" << endl;
    cout << setw(30) << "  -v, --version"          << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help"             << setw(50) << "print this help" << endl;
    cout << endl;
    // clang-format on
}
