#include <iostream>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "DynamicConfiguration.hh"
#include "CtrOptMan.hh"
#include "CtrCfgMap.hh"
#include "CtrConfig.hh"
#include "CtrSrvMan.hh"

using namespace std;
using namespace diffraflow;

DynamicConfiguration* gZookeeperConfClient = nullptr;
CtrConfig* gConfiguration = nullptr;
CtrSrvMan* gServerManager = nullptr;

int execute_zk_actions(DynamicConfiguration* zk_conf_client, vector<string> zk_actions) {
    for (size_t i = 0; i < zk_actions.size(); i++) {
        string action = zk_actions[i];
        size_t sepidx = action.find("#");
        string op = action.substr(0, sepidx);
        string oprnd = action.substr(sepidx + 1);
        string znode, conf_map_file;
        CtrCfgMap conf_map;
        if (op == "C" || op == "U") {
            size_t colon = oprnd.find(":");
            if (colon == string::npos || colon == 0 || colon == oprnd.length() - 1) {
                cerr << "operator '" << op << "' with wrong operand '" << oprnd << "'." << endl;
                return 1;
            }
            znode = oprnd.substr(0, colon);
            conf_map_file = oprnd.substr(colon + 1);
            if (conf_map.load(conf_map_file.c_str())) {
                cout << "successfully loaded config map file: " << conf_map_file << endl;
            } else {
                cout << "failed to load config map file: " << conf_map_file << endl;
                return 1;
            }
        } else {
            znode = oprnd;
        }
        // run the action
        if (op == "C") {
            if (ZOK == zk_conf_client->zookeeper_create_config(znode.c_str(), conf_map.data)) {
                cout << "successfully created znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
            } else {
                cout << "failed to create znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
                return 1;
            }
        } else if (op == "U") {
            if (ZOK == zk_conf_client->zookeeper_change_config(znode.c_str(), conf_map.data)) {
                cout << "successfully updated znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
            } else {
                cout << "failed to update znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
                return 1;
            }
        } else if (op == "R") {
            int version;
            if (ZOK == zk_conf_client->zookeeper_fetch_config(znode.c_str(), conf_map.data, conf_map.mtime, version)) {
                cout << "znode: " << znode << endl;
                conf_map.print();
            } else {
                cout << "failed to read the data of znode " << znode << "." << endl;
                return 1;
            }
        } else if (op == "L") {
            vector<string> children_list;
            if (ZOK == zk_conf_client->zookeeper_get_children(znode.c_str(), children_list)) {
                if (children_list.empty()) {
                    cout << "znode " << znode << " has no children." << endl;
                } else {
                    cout << "znode: " << znode << endl;
                    for (size_t i = 0; i < children_list.size(); i++) {
                        cout << " | - " << children_list[i] << endl;
                    }
                }
            } else {
                cout << "failed to get the children of znode " << znode << "." << endl;
                return 1;
            }
        } else if (op == "D") {
            if (ZOK == zk_conf_client->zookeeper_delete_config(znode.c_str())) {
                cout << "successfully deleted znode " << znode << "." << endl;;
            } else {
                cout << "failed to delete znode " << znode << "." << endl;
                return 1;
            }
        }
    }
    return 0;
}

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
    if (gZookeeperConfClient != nullptr) {
        delete gZookeeperConfClient;
        gZookeeperConfClient = nullptr;
    }
    exit(0);
}

void init(CtrConfig* config_obj) {
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
    if (!option_man.zk_conf_file.empty()) {
        gZookeeperConfClient = new DynamicConfiguration();
        if (!gZookeeperConfClient->load(option_man.zk_conf_file.c_str())) {
            cerr << "failed to load zookeeper client configuration file." << endl;
            return 1;
        }
        gZookeeperConfClient->zookeeper_print_setting();
        if (!gZookeeperConfClient->zookeeper_start(true)) {
            cerr << "failed to start zookeeper session." << endl;
            return 1;
        }
    }

    // execute zookeeper actions if have
    if (option_man.zk_actions.size() > 0) {
        if (gZookeeperConfClient == nullptr) {
            cout << "zookeeper session is not started, all zk operations are ignored." << endl;
        } else {
            int result = execute_zk_actions(gZookeeperConfClient, option_man.zk_actions);
            if (result > 0) {
                return result;
            }
        }
    }

    gConfiguration = new CtrConfig();
    if (!option_man.config_file.empty() && !gConfiguration->load(option_man.config_file.c_str())) {
        cout << "Failed to load configuration file: " << option_man.config_file << endl;
        return 1;
    }
    gConfiguration->print();

    // start servers if needed
    // ------------------------------------------------------
    init(gConfiguration);

    if (!gConfiguration->http_host.empty()) {
        if (!option_man.monaddr_file.empty()) {
            gServerManager = new CtrSrvMan(gConfiguration, option_man.monaddr_file.c_str(), gZookeeperConfClient);
        } else {
            gServerManager = new CtrSrvMan(gConfiguration, nullptr, gZookeeperConfClient);
        }
        gServerManager->start_run();
    }

    clean(0);
    // ------------------------------------------------------

    return 0;
}
