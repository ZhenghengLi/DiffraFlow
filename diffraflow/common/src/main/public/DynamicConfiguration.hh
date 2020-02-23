#ifndef __DynamicConfiguration_H__
#define __DynamicConfiguration_H__

#include "GenericConfiguration.hh"
#include <map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>
#include <zookeeper/zookeeper.h>

using std::map;
using std::string;
using std::mutex;
using std::atomic;
using std::atomic_bool;
using std::atomic_int;

namespace diffraflow {

    class DynamicConfiguration: public GenericConfiguration {
    public:
        DynamicConfiguration();
        ~DynamicConfiguration();

        virtual bool load(const char* filename);
        virtual void print();

        // convert the key-values in conf_map_ to the field values of sub-class;
        virtual void convert_and_check();

    protected:
        map<string, string> conf_map_;
        mutex conf_map_mtx_;

    public:
        // zookeeper operations
        void zookeeper_start(bool is_upd);
        void zookeeper_stop();
        bool zookeeper_create_config(string parent_node,
            const map<string, string>& config_map);
        bool zookeeper_change_config(string parent_node,
            const map<string, string>& config_map);
        bool zookeeper_watch_config(string parent_node,
            DynamicConfiguration* config_obj);

    private:
        // zookeeper callbacks
        static void zookeeper_main_watcher_(
            zhandle_t* zh, int type, int state, const char* path, void* context
        );

    private:
        // zookeeper configurations
        zhandle_t* zookeeper_handle_;
        string zookeeper_server_;
        string zookeeper_root_node_;
        string zookeeper_log_level_;
        int zookeeper_expiration_time_;
        string zookeeper_auth_string_;  // user:password
        atomic_bool zookeeper_initialized_;
        atomic_bool zookeeper_is_updater_;
        atomic_int count_down_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

}

#endif