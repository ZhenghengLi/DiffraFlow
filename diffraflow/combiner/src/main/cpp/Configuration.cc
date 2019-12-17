#include "Configuration.hh"
#include <iostream>
#include <fstream>
#include <sstream>

using std::ifstream;
using std::stringstream;
using std::cout;
using std::cerr;
using std::endl;

diffraflow::Configuration::Configuration() {

}

diffraflow::Configuration::~Configuration() {

}

bool diffraflow::Configuration::load(const char* filename) {
    ifstream config_file;
    config_file.open(filename);
    if (!config_file.is_open()) {
        cout << "config file open failed." << endl;
        return false;
    }
    stringstream ss;
    string oneline;
    string key, value, sep;
    while (true) {
        key = ""; value = ""; sep = ""; oneline = "";
        getline(config_file, oneline);
        if (config_file.eof()) break;
        // skip comments
        if (oneline.find("#") != string::npos) continue;
        // read key-value
        ss.clear(); ss.str(oneline);
        ss >> key >> sep >> value;
        // skip invalid line
        if (key == "" || value == "" || sep != "=") continue;
        // use key-value from here
        if (key == "port") {
            port = atoi(value.c_str());
        } else {
            cerr << "ERROR: found unknown configuration: " << key << endl;
            return false;
        }
    }
    config_file.close();
    return true;
}

void diffraflow::Configuration::print() {
    cout << " = Configuration Dump Begin =" << endl;
    cout << "  port = " << port << endl;
    cout << " = Configuration Dump End =" << endl;

}
