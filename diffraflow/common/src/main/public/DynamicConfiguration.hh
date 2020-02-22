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
        virtual void convert();

    protected:
        map<string, string> conf_map_;
        mutex conf_map_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;

    /////////////////////////////////////////////////////////////
    // for ZooKeeper
    /////////////////////////////////////////////////////////////
    public:
        static void zookeeper_start(DynamicConfiguration* obj, bool is_upd);
        static void zookeeper_stop();
        // create config nodes
        static bool zookeeper_create_config(string parent_node,
            const map<string, string>& cfg_map);
        // update the data in config nodes
        static bool zookeeper_change_config(string parent_node,
            const map<string, string>& cfg_map);

    private:
        static void zookeeper_start_session_();
        static void zookeeper_main_watcher_(
            zhandle_t* zh, int type, int state, const char* path, void* context
        );

    private:
        static atomic<DynamicConfiguration*> the_obj_;
        static zhandle_t* zookeeper_handle_;
        static string zookeeper_server_;
        static string zookeeper_root_node_;
        static string zookeeper_log_level_;
        static int zookeeper_expiration_time_;
        static string zookeeper_auth_string_;  // user:password
        static atomic_bool zookeeper_intialized_;
        static atomic_bool zookeeper_is_updater_;
        static atomic_int count_down_;

    };

}

#endif