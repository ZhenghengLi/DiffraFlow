#ifndef GenericConfiguration_H
#define GenericConfiguration_H

#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <log4cxx/logger.h>

using std::vector;
using std::list;
using std::pair;
using std::string;

namespace diffraflow {
    class GenericConfiguration {
    public:
        GenericConfiguration();
        virtual ~GenericConfiguration();

        virtual bool load(const char* filename) = 0;
        virtual void print() = 0;

    protected:
        bool read_conf_KV_list_(const char* filename, list<pair<string, string>>& conf_KV_list);

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif