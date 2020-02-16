#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "CmbOptMan.hh"
#include "CmbConfig.hh"
#include "CmbSrvMan.hh"

using namespace std;
using namespace diffraflow;


CmbConfig* gConfiguration = nullptr;
CmbSrvMan* gServerManager = nullptr;

void clean(int signum) {
    cout << "do cleaning ..." << endl;
    if (gServerManager != nullptr) {
        gServerManager->terminate();
        delete gServerManager;
        gServerManager = nullptr;
        cout << "Server is terminated." << endl;
    }
    if (gConfiguration != nullptr) {
        delete gConfiguration;
        gConfiguration = nullptr;
    }
    exit(0);
}

void init(CmbConfig* config_obj) {
    // register signal action
    struct sigaction action;
    // for Ctrl-C
    memset(&action, 0, sizeof(action));
    action.sa_handler = &clean;
    if (sigaction(SIGINT, &action, nullptr)) {
        perror("sigaction() failed.");
        exit(1);
    }
    // ignore SIGPIPE
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &action, nullptr)) {
        perror("sigaction() failed.");
        exit(1);
    }
}

int main(int argc, char** argv) {
    // process command line parameters
    CmbOptMan option_man;
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
    gConfiguration = new CmbConfig();
    if (!gConfiguration->load(option_man.config_file.c_str())) {
        cout << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }
    gConfiguration->print();

    // ------------------------------------------------------
    init(gConfiguration);

    gServerManager = new CmbSrvMan(gConfiguration);
    gServerManager->start_run();

    clean(0);
    // ------------------------------------------------------

    return 0;
}
