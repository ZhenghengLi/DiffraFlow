#ifndef __MetricsProvider_H__
#define __MetricsProvider_H__

#include <cpprest/json.h>

using namespace web;

namespace diffraflow {
    struct MetricsProvider {
        virtual json::value collect_metrics() = 0;
    };
}

#endif