#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "IngOptMan.hh"
#include "IngConfig.hh"
#include "IngPipeline.hh"

using namespace std;
using namespace diffraflow;

IngConfig* gConfiguration = nullptr;
IngPipeline* gPipeline = nullptr;

void clean(int signum) {
    cout << "do cleaning ..." << endl;
    if (gPipeline != nullptr) {
        gPipeline->terminate();
        delete gPipeline;
        gPipeline = nullptr;
        cout << "Pipeline is terminated." << endl;
    }
    if (gConfiguration != nullptr) {
        delete gConfiguration;
        gConfiguration = nullptr;
    }
    exit(0);
}

void init(IngConfig* config_obj) {
    // register signal action
    struct sigaction action;
    // for Ctrl-C
    memset(&action, 0, sizeof(action));
    action.sa_handler = &clean;
    if (sigaction(SIGINT, &action, nullptr)) {
        perror("sigaction() failed for SIGINT.");
        exit(1);
    }
    // Kubernetes uses SIGTERM to terminate Pod
    if (sigaction(SIGTERM, &action, nullptr)) {
        perror("sigaction() failed for SIGTERM.");
        exit(1);
    }
    // ignore SIGPIPE
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &action, nullptr)) {
        perror("sigaction() failed for ignoring SIGPIPE.");
        exit(1);
    }
}

int main(int argc, char** argv) {
    // process command line parameters
    IngOptMan option_man;
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
    gConfiguration = new IngConfig();
    if (!gConfiguration->load(option_man.config_file.c_str())) {
        cout << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }

    gConfiguration->print();
    if (gConfiguration->zookeeper_setting_is_ready()) {
        cout << "Connectiong to ZooKeeper ..." << endl;
        gConfiguration->zookeeper_start();
        cout << "start to sync configurations with zookeeper ..." << endl;
        gConfiguration->zookeeper_sync_config();
        gConfiguration->zookeeper_sync_wait();
    }

    // ------------------------------------------------------
    init(gConfiguration);

    gPipeline = new IngPipeline(gConfiguration);
    gPipeline->start_run();

    clean(0);
    // ------------------------------------------------------

    return 0;
}
