#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <chrono>
#include <thread>
#include <future>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "TrgOptMan.hh"
#include "TrgCoordinator.hh"

using namespace std;
using namespace diffraflow;

using chrono::microseconds;
using chrono::duration;
using chrono::system_clock;

TrgCoordinator* gTriggerCoordinator = nullptr;

void clean(int signum) {
    cout << "do cleaning ..." << endl;
    if (gTriggerCoordinator != nullptr) {
        gTriggerCoordinator->delete_trigger_clients();
        delete gTriggerCoordinator;
        gTriggerCoordinator = nullptr;
        cout << "trigger coordinator is stopped." << endl;
    }
    exit(0);
}

void init() {
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
    TrgOptMan option_man;
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

    init();

    gTriggerCoordinator = new TrgCoordinator();
    if (!gTriggerCoordinator->create_trigger_clients(option_man.sender_list_file.c_str(), option_man.sender_id)) {
        cout << "failed to create trigger clients from file " << option_man.sender_list_file << endl;
        gTriggerCoordinator->delete_trigger_clients();
        return 1;
    }

    if (option_man.start_event_index >= 0 && option_man.event_count > 0 && option_man.interval_microseconds >= 0) {
        async([&]() {
            gTriggerCoordinator->trigger_many_events(
                option_man.start_event_index, option_man.event_count, option_man.interval_microseconds);
        }).wait();
    } else {
        int event_index;
        while (true) {
            cout << "input event index: " << flush;
            cin >> event_index;
            if (event_index < 0) {
                cout << "event index is less than zero." << endl;
                continue;
            }
            duration<double, micro> start_time = system_clock::now().time_since_epoch();
            bool succ_flag = false;
            async([&]() { succ_flag = gTriggerCoordinator->trigger_one_event(event_index); }).wait();
            duration<double, micro> finish_time = system_clock::now().time_since_epoch();
            long time_used = finish_time.count() - start_time.count();
            if (succ_flag) {
                cout << "successfully triggered event " << event_index << " using " << time_used << " microseconds."
                     << endl;
            } else {
                cout << "failed to trigger event " << event_index << " after " << time_used << " microseconds." << endl;
            }
        }
    }

    clean(0);

    return 0;
}
