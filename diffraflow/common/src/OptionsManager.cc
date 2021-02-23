#include "OptionsManager.hh"
#include "VersionInfo.hh"
#include <iostream>

using std::cout;
using std::endl;

diffraflow::OptionsManager::OptionsManager(const char* sw_name) {
    software_name_ = sw_name;
    version_flag_ = false;
}

diffraflow::OptionsManager::~OptionsManager() {}

void diffraflow::OptionsManager::print() {
    if (version_flag_) {
        print_version_();
    } else {
        print_help_();
    }
}

void diffraflow::OptionsManager::print_version_() {
    cout << endl;
    cout << "    " << software_name_ << " - "
         << "DiffraFlow Project" << endl;
    cout << "    " << VERSION_STRING << " (" << VERSION_DATE << ", compiled " << __DATE__ << " " << __TIME__ << ")"
         << endl;
    cout << endl;
    cout << COPYRIGHT_STATEMENT << endl;
    cout << endl;
    cout << " Main Contributors: " << endl;
    cout << MAIN_CONTRIBUTORS << endl;

    cout << endl;
}
