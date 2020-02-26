#include <iostream>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/logmanager.h>
#include <log4cxx/logstring.h>

#include "DynamicConfiguration.hh"
#include "CtrOptMan.hh"
#include "CtrCfgMap.hh"

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

    for (size_t i = 0; i < option_man.zk_actions.size(); i++) {
        string action = option_man.zk_actions[i];
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
            if (zk_conf_client->zookeeper_create_config(znode.c_str(), conf_map.data)) {
                cout << "successfully created znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
            } else {
                cout << "failed to create znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
                return 1;
            }
        } else if (op == "U") {
            if (zk_conf_client->zookeeper_change_config(znode.c_str(), conf_map.data)) {
                cout << "successfully updated znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
            } else {
                cout << "failed to update znode " << znode
                     << " with config map file " << conf_map_file << "." << endl;
                return 1;
            }
        } else if (op == "R") {
            if (zk_conf_client->zookeeper_fetch_config(znode.c_str(), conf_map.data, conf_map.mtime)) {
                cout << "znode: " << znode << endl;
                conf_map.print();
            } else {
                cout << "failed to read the data of znode " << znode << "." << endl;
                return 1;
            }
        } else if (op == "L") {
            vector<string> children_list;
            if (zk_conf_client->zookeeper_get_children(znode.c_str(), children_list)) {
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
            if (zk_conf_client->zookeeper_delete_config(znode.c_str())) {
                cout << "successfully deleted znode " << znode << "." << endl;;
            } else {
                cout << "failed to delete znode " << znode << "." << endl;
                return 1;
            }
        }
    }

    delete zk_conf_client;
    zk_conf_client = nullptr;

    return 0;
}
