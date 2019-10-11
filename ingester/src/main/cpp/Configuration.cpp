#include "Configuration.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using std::ifstream;
using std::stringstream;
using std::cout;
using std::endl;

shine::Configuration::Configuration() {

}

shine::Configuration::~Configuration() {

}

bool shine::Configuration::load(const char* filename) {
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
        if (key.find("kafka.") == 0) {
            string name = key.substr(6);
            kafka_props[name] = value;
        }
    }
    config_file.close();
    return true;
}

void shine::Configuration::print() {
    cout << " = Configuration Dump Begin =" << endl;
    cout << endl;
    cout << "   kafka: " << endl;
    for (map<string, string>::iterator iter = kafka_props.begin(); iter != kafka_props.end(); iter++) {
        cout << "     " << iter->first << " = " << iter->second << endl;
    }
    cout << endl;
    cout << " = Configuration Dump End =" << endl;

}
