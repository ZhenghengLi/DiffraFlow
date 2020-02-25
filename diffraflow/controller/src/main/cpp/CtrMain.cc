#include <iostream>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "CtrOptMan.hh"
#include "DynamicConfiguration.hh"

using namespace std;
using namespace diffraflow;

int main(int argc, char** argv) {
    // process command line parameters
    CtrOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }
    return 0;
}
