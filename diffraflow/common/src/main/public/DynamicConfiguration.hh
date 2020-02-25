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
        int64_t conf_map_mtime_;
        mutex conf_map_mtx_;

    public:
        // zookeeper operations
        bool zookeeper_start(bool is_upd);
        bool zookeeper_start();
        void zookeeper_stop();
        // for updater
        bool zookeeper_create_config(const char* config_path,
            const map<string, string>& config_map);
        bool zookeeper_delete_config(const char* config_path);
        bool zookeeper_change_config(const char* config_path,
            const map<string, string>& config_map);
        // for reader
        bool zookeeper_sync_config();
        // print setting
        void zookeeper_print_setting();

    private:
        void zookeeper_connection_wait_();
        bool zookeeper_authadding_wait_();

    private:
        // zookeeper callbacks
        static void zookeeper_main_watcher_(zhandle_t* zh, int type,
            int state, const char* path, void* context);
        static void zookeeper_config_watcher_(zhandle_t* zh, int type,
            int state, const char* path, void* context);
        static void zookeeper_auth_completion_(int rc, const void* data);
        static void zookeeper_data_completion_(int rc, const char *value,
            int value_len, const struct Stat *stat, const void *data);

    private:
        enum CallbackRes_ {kUnknown, kSucc, kFail};

        // zookeeper configurations
        zhandle_t*              zookeeper_handle_;
        string                  zookeeper_server_;
        string                  zookeeper_chroot_;
        string                  zookeeper_config_path_;
        string                  zookeeper_log_level_;
        int                     zookeeper_expiration_time_;
        string                  zookeeper_auth_string_;  // user:password
        atomic_bool             zookeeper_is_updater_;

        // zookeeper connectiong status
        bool                    zookeeper_connected_;
        mutex                   zookeeper_connected_mtx_;
        condition_variable      zookeeper_connected_cv_;

        // zookeeper operation results
        CallbackRes_            zookeeper_auth_res_;
        mutex                   zookeeper_auth_res_mtx_;
        condition_variable      zookeeper_auth_res_cv_;

        CallbackRes_            zookeeper_data_res_;
        mutex                   zookeeper_data_res_mtx_;
        condition_variable      zookeeper_data_res_cv_;
        string                  zookeeper_data_string_;
        int64_t                 zookeeper_data_mtime_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

}

#endif