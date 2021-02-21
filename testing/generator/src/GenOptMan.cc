#include "GenOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::GenOptMan::opt_string_[] = "d:i:o:c:a:e:m:vh";

// clang-format off
const option diffraflow::GenOptMan::long_opts_[] = {
    {"datdir",  required_argument, NULL, 'd'},
    {"modid",   required_argument, NULL, 'i'},
    {"outdir",  required_argument, NULL, 'o'},
    {"calib",   required_argument, NULL, 'c'},
    {"align",   required_argument, NULL, 'a'},
    {"event",   required_argument, NULL, 'e'},
    {"maxevt",  required_argument, NULL, 'm'},
    {"help",    no_argument,       NULL, 'h'},
    {"version", no_argument,       NULL, 'v'},
    {NULL,      no_argument,       NULL, 0}
};
// clang-format on

diffraflow::GenOptMan::GenOptMan() : OptionsManager("generator") {
    data_dir.clear();
    module_id = 0;
    output_dir.clear();
    calib_file.clear();
    align_file.clear();
    event_file.clear();
    max_events = 10000;
}

diffraflow::GenOptMan::~GenOptMan() {}

bool diffraflow::GenOptMan::parse(int argc, char** argv) {
    while (true) {
        int opt, idx;
        opt = getopt_long(argc, argv, opt_string_, long_opts_, &idx);
        if (opt < 0) break;
        switch (opt) {
        case 'd':
            data_dir = optarg;
            break;
        case 'i':
            module_id = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'c':
            calib_file = optarg;
            break;
        case 'a':
            align_file = optarg;
            break;
        case 'e':
            event_file = optarg;
            break;
        case 'm':
            max_events = atoi(optarg);
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
    if (data_dir.empty()) {
        cerr << "data directory is not set." << endl;
        succ_flag = false;
    }
    if (output_dir.empty()) {
        cerr << "output directory is not set." << endl;
        succ_flag = false;
    }
    if (module_id < 0 || module_id > 15) {
        cerr << "module ID is out of range (0 -- 15)." << endl;
        succ_flag = false;
    }
    if (calib_file.empty()) {
        cerr << "calibraion file is not set." << endl;
        succ_flag = false;
    }
    if (align_file.empty()) {
        cerr << "alignment file is not set." << endl;
        succ_flag = false;
    }
    if (event_file.empty()) {
        cerr << "event number file is not set." << endl;
        succ_flag = false;
    }
    if (max_events < 10) {
        cerr << "maximum events per file is too small." << endl;
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

void diffraflow::GenOptMan::print_help_() {
    // clang-format off
    cout << "Usage:" << endl;
    cout << "  " << software_name_ << " [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -d, --datdir=DIR"    << setw(50) << "data directory" << endl;
    cout << setw(30) << "  -i, --modid=ID"      << setw(50) << "module ID (0 -- 15)" << endl;
    cout << setw(30) << "  -o, --outdir=DIR"    << setw(50) << "output directory" << endl;
    cout << setw(30) << "  -c, --calib=FILE"    << setw(50) << "calibration file" << endl;
    cout << setw(30) << "  -a, --align=FILE"    << setw(50) << "alignment file" << endl;
    cout << setw(30) << "  -e, --event=FILE"    << setw(50) << "event num file" << endl;
    cout << setw(30) << "  -m, --maxevt=NUM"    << setw(50) << "maximum events per file" << endl;
    cout << setw(30) << "  -v, --version"       << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help"          << setw(50) << "print this help" << endl;
    cout << endl;
    // clang-format on
}
