#ifndef __DynamicConfiguration_H__
#define __DynamicConfiguration_H__

#include "GenericConfiguration.hh"
#include <map>
#include <string>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>
#include <zookeeper/zookeeper.h>

using std::map;
using std::string;
using std::mutex;
using std::condition_variable;
using std::thread;
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
        // for updater
        bool zookeeper_create_config(string config_node,
            const map<string, string>& config_map);
        bool zookeeper_change_config(string config_node,
            const map<string, string>& config_map);
        // for reader
        bool zookeeper_watch_config(string config_node,
            DynamicConfiguration* config_obj = nullptr);

    private:
        // zookeeper callbacks
        static void zookeeper_main_watcher_(zhandle_t* zh, int type,
            int state, const char* path, void* context);
        static void zookeeper_auth_completion_(int rc, const void* data);

    private:
        enum CallbackRes_ {kUnknown, kSucc, kFail};

        // zookeeper configurations
        zhandle_t*          zookeeper_handle_;
        string              zookeeper_server_;
        string              zookeeper_root_node_;
        string              zookeeper_log_level_;
        int                 zookeeper_expiration_time_;
        string              zookeeper_auth_string_;  // user:password
        atomic_bool         zookeeper_is_updater_;

        // zookeeper connectiong status
        bool                zookeeper_connected_;
        mutex               zookeeper_connected_mtx_;
        condition_variable  zookeeper_connected_cv_;

        // zookeeper operation results
        CallbackRes_        zookeeper_auth_res_;
        mutex               zookeeper_auth_res_mtx_;
        condition_variable  zookeeper_auth_res_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

}

#endif