#include <iostream>

#include "IngOptMan.hh"

using namespace std;
using namespace diffraflow;

int main(int argc, char** argv) {
    // process command line parameters
    IngOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }


    return 0;
}
