#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <log4cxx/basicconfigurator.h>
#include <log4cxx/propertyconfigurator.h>

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

    if (argc < 2) {
        cout << "Usage: " << "combiner" << " <config.conf>" << endl;
        return 2;
    }
    string config_fn = argv[1];

    log4cxx::BasicConfigurator::configure();
    // TODO: log4cxx::PropertyConfigurator::configure(filename)

    gConfiguration = new CmbConfig();
    if (!gConfiguration->load(config_fn.c_str())) {
        cout << "Failed to load configuration file: " << config_fn << endl;
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
