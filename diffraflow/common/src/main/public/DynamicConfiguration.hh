#ifndef __DynamicConfiguration_H__
#define __DynamicConfiguration_H__

#include "GenericConfiguration.hh"
#include <map>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>

using std::map;
using std::string;
using std::mutex;
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
        static bool config_zookeeper(DynamicConfiguration* obj);

    private:
        static DynamicConfiguration* the_obj_;
        static string zookeeper_server_;
        static int zookeeper_expiration_time_;
        static string zookeeper_auth_string_;  // user:password
        static atomic_bool zookeeper_intialized_;
        static atomic_int count_down_;

    };

}

#endif