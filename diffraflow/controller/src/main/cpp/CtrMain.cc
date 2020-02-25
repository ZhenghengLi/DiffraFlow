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
    // set up zk client
    DynamicConfiguration* zk_conf_client = new DynamicConfiguration();
    if (!zk_conf_client->load(option_man.zk_conf_file.c_str())) {
        cerr << "failed to load zookeeper client configuration file." << endl;
        return 1;
    }
    zk_conf_client->zookeeper_print_setting();
    if (!zk_conf_client->zookeeper_start(true)) {
        cerr << "failed to set zookeeper session." << endl;
        return 1;
    }



    delete zk_conf_client;
    zk_conf_client = nullptr;

    return 0;
}
