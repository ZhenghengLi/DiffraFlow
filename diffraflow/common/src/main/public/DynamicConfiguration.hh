#ifndef __DynamicConfiguration_H__
#define __DynamicConfiguration_H__

#include "GenericConfiguration.hh"
#include "MetricsProvider.hh"
#include <map>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>
#include <zookeeper/zookeeper.h>
#include <ctime>

using std::map;
using std::vector;
using std::string;
using std::mutex;
using std::condition_variable;
using std::thread;
using std::atomic;
using std::atomic_bool;
using std::atomic_int;

namespace diffraflow {

    class DynamicConfiguration: public GenericConfiguration, public MetricsProvider {
    public:
        DynamicConfiguration();
        virtual ~DynamicConfiguration();

        virtual bool load(const char* filename);
        virtual void print();

        json::value collect_metrics() override;

    protected:
        // convert the key-values in conf_map to the field values of sub-class;
        virtual bool check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime);

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
        bool zookeeper_fetch_config(const char* config_path,
            map<string, string>& config_map, time_t& config_mtime);
        bool zookeeper_get_children(const char* config_path,
            vector<string>& children_list);
        // for reader
        void zookeeper_sync_config();
        // print setting
        void zookeeper_print_setting();

    protected:
        // parse setting
        bool zookeeper_parse_setting_(list< pair<string, string> > conf_KV_list);

    private:
        void zookeeper_connection_wait_();
        bool zookeeper_authadding_wait_();
        void zookeeper_add_auth_();
        void zookeeper_get_config_();
        bool zookeeper_check_path_(const char* path);

    private:
        // zookeeper callbacks
        static void zookeeper_main_watcher_(zhandle_t* zh, int type,
            int state, const char* path, void* context);
        static void zookeeper_config_watcher_(zhandle_t* zh, int type,
            int state, const char* path, void* context);
        static void zookeeper_auth_completion_(int rc, const void* data);
        static void zookeeper_data_completion_(int rc, const char* value,
            int value_len, const struct Stat* stat, const void* data);
        static void zookeeper_stat_completion_(int rc,
            const struct Stat* stat, const void* data);

    private:
        enum CallbackRes_ {kUnknown, kSucc, kFail};

        // zookeeper configurations
        zhandle_t*              zookeeper_handle_;
        string                  zookeeper_server_;
        string                  zookeeper_chroot_;
        string                  zookeeper_log_level_;
        int                     zookeeper_expiration_time_;
        string                  zookeeper_auth_string_;  // user:password
        string                  zookeeper_config_path_;
        atomic_bool             zookeeper_is_updater_;

        // znode buffer for fetching
        mutex                   zookeeper_znode_mtx_;
        char*                   zookeeper_znode_buffer_;
        int                     zookeeper_znode_buffer_cap_;
        int                     zookeeper_znode_buffer_len_;
        Stat                    zookeeper_znode_stat_;

        // zookeeper connectiong status
        bool                    zookeeper_connected_;
        mutex                   zookeeper_connected_mtx_;
        condition_variable      zookeeper_connected_cv_;

        // zookeeper operation results
        CallbackRes_            zookeeper_auth_res_;
        mutex                   zookeeper_auth_res_mtx_;
        condition_variable      zookeeper_auth_res_cv_;

        // for metrics
        json::value             zookeeper_config_json_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

}

#endif
