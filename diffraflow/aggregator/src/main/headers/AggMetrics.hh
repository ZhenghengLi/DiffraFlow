#ifndef __AggMetrics_H__
#define __AggMetrics_H__

#include <string>
#include <thread>
#include <pulsar/Client.h>
#include <pulsar/Consumer.h>
#include <pulsar/Message.h>
#include <pulsar/Result.h>
#include <log4cxx/logger.h>

using std::string;
using std::thread;

namespace diffraflow {
    class AggMetrics {
    public:
        explicit AggMetrics(string pulsar_url, int threads_count = 1);
        ~AggMetrics();

    private:
        pulsar::Client* pulsar_client_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif