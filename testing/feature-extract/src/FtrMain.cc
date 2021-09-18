#include <iostream>
#include <fstream>
#include <cmath>
#include <H5Cpp.h>
#include <stdio.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "FtrOptMan.hh"
#include "FtrConfig.hh"

using namespace diffraflow;
using namespace std;

int main(int argc, char** argv) {
    // process command line parameters
    FtrOptMan option_man;
    if (!option_man.parse(argc, argv)) {
        option_man.print();
        return 2;
    }
    // configure logger
    if (option_man.logconf_file.empty()) {
        log4cxx::LogManager::getLoggerRepository()->setConfigured(true);
        static const log4cxx::LogString logfmt(LOG4CXX_STR("%d [%t] %-5p %c - %m%n"));
        log4cxx::LayoutPtr layout(new log4cxx::PatternLayout(logfmt));
        log4cxx::AppenderPtr appender(new log4cxx::ConsoleAppender(layout));
        log4cxx::Logger::getRootLogger()->addAppender(appender);
        log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());
    } else {
        log4cxx::PropertyConfigurator::configure(option_man.logconf_file);
    }
    // parse configuration file
    FtrConfig* config = new FtrConfig();
    if (!option_man.config_file.empty() && !config->load(option_man.config_file.c_str())) {
        cout << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }
    config->print();

    // ===== process begin =======================================================================

    // ===== process end =========================================================================

    // clean
    delete config;
    config = nullptr;

    return 0;
}
