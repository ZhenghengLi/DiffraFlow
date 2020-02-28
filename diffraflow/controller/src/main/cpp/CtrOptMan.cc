#include "CtrOptMan.hh"
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::left;
using std::setw;

const char diffraflow::CtrOptMan::opt_string_[] = "c:z:l:C:D:U:R:L:vh";

const option diffraflow::CtrOptMan::long_opts_[] = {
    {"config",      required_argument,  NULL, 'c'},
    {"zkconfig",    required_argument,  NULL, 'z'},
    {"logconf",     required_argument,  NULL, 'l'},
    {"zkcreate",    required_argument,  NULL, 'C'},
    {"zkdelete",    required_argument,  NULL, 'D'},
    {"zkupdate",    required_argument,  NULL, 'U'},
    {"zkread",      required_argument,  NULL, 'R'},
    {"zklist",      required_argument,  NULL, 'L'},
    {"help",        no_argument,        NULL, 'h'},
    {"version",     no_argument,        NULL, 'v'},
    {NULL,          no_argument,        NULL, 0}
};

diffraflow::CtrOptMan::CtrOptMan(): OptionsManager("controller") {

}

diffraflow::CtrOptMan::~CtrOptMan() {

}

bool diffraflow::CtrOptMan::parse(int argc, char** argv) {
    zk_actions.clear();
    while (true) {
        int opt, idx;
        opt = getopt_long(argc, argv, opt_string_, long_opts_, &idx);
        if (opt < 0) break;
        switch (opt) {
        case 'c':
            config_file = optarg;
            break;
        case 'z':
            zk_conf_file = optarg;
            break;
        case 'l':
            logconf_file = optarg;
            break;
        case 'C':
            zk_actions.push_back(string("C#") + optarg);
            break;
        case 'D':
            zk_actions.push_back(string("D#") + optarg);
            break;
        case 'U':
            zk_actions.push_back(string("U#") + optarg);
            break;
        case 'R':
            zk_actions.push_back(string("R#") + optarg);
            break;
        case 'L':
            zk_actions.push_back(string("L#") + optarg);
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
    // if (config_file.empty()) {
    //     cerr << "configuration file should be set." << endl;
    //     succ_flag = false;
    // }
    if (zk_conf_file.empty()) {
        cerr << "zookeeper client config file should be set." << endl;
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

void diffraflow::CtrOptMan::print_help_() {
    cout << "Usage:" << endl;
    cout << "  " << software_name_ << " [OPTION...]" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << left;
    cout << setw(30) << "  -c, --config=FILE"           << setw(50) << "configuration file" << endl;
    cout << setw(30) << "  -z, --zkconfig=FILE"         << setw(50) << "zookeeper client configuration file" << endl;
    cout << setw(30) << "  -l, --logconf=FILE"          << setw(50) << "log configuration file" << endl;
    cout << setw(30) << "  -C, --zkcreate=ZNODE:FILE"   << setw(50) << "create a znode with confmap file" << endl;
    cout << setw(30) << "  -U, --zkupdate=ZNODE:FILE"   << setw(50) << "update a znode with confmap file" << endl;
    cout << setw(30) << "  -R, --zkread=ZNODE"          << setw(50) << "read and print a znode" << endl;
    cout << setw(30) << "  -L, --zklist=ZNODE"          << setw(50) << "list the children of a znode" << endl;
    cout << setw(30) << "  -D, --zkdelete=ZNODE"        << setw(50) << "delete a znode" << endl;
    cout << setw(30) << "  -v, --version"               << setw(50) << "print version and copyright" << endl;
    cout << setw(30) << "  -h, --help"                  << setw(50) << "print this help" << endl;
    cout << endl;
}
