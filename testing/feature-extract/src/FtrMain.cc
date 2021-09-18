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
#include "ImageFileHDF5R.hh"

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
        cerr << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }
    config->print();

    // ===== process begin =======================================================================
    ImageFileHDF5R image_file(10, false);
    if (!image_file.open(option_man.data_file.c_str())) {
        cerr << "Failed to open image data file: " << option_man.data_file << endl;
        return 1;
    }
    // cout << "create time: " << image_file.create_time() << endl;
    ofstream outfile;
    if (!option_man.output_file.empty()) {
        outfile.open(option_man.output_file.c_str());
        if (!outfile.is_open()) {
            cerr << "Failed to open output file: " << option_man.output_file << endl;
            return 1;
        }
    }

    ImageDataField image_data;
    while (image_file.next_batch()) {
        while (image_file.next_image(image_data)) {
            //
            cout << image_file.current_position() << endl;
        }
    }

    if (outfile.is_open()) {
        outfile.close();
    }
    image_file.close();

    // ===== process end =========================================================================

    // clean
    delete config;
    config = nullptr;

    return 0;
}
