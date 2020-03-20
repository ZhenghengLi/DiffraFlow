#ifndef __MetricsProvider_H__
#define __MetricsProvider_H__

#include <jsoncpp/json/json.h>

namespace diffraflow {
    struct MetricsProvider {
        virtual Json::Value collect() = 0;
    };
}

#endif