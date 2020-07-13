#include <iostream>
#include "GenOptMan.hh"

using namespace std;
using namespace diffraflow;

int main(int argc, char** argv) {
    GenOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }

    cout << "data_dir: " << option_man.data_dir << endl;
    cout << "module_id: " << option_man.module_id << endl;
    cout << "output_dir: " << option_man.output_dir << endl;
    cout << "calib_file: " << option_man.calib_file << endl;
    cout << "align_file: " << option_man.align_file << endl;
    cout << "event_file: " << option_man.event_file << endl;
    cout << "max_events: " << option_man.max_events << endl;

    return 0;
}
