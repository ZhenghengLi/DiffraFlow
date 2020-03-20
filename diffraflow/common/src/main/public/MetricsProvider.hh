#ifndef __MetricsProvider_H__
#define __MetricsProvider_H__

#include <jsoncpp/json/json.h>

namespace diffraflow {
    struct MetricsProvider {
        virtual Json::Value collect_metrics() = 0;
    };
}

#endif