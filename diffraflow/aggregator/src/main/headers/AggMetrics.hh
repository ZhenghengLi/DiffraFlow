#ifndef __AggMetrics_H__
#define __AggMetrics_H__

#include <string>
#include <thread>
#include <pulsar/Client.h>
#include <pulsar/Consumer.h>
#include <pulsar/Message.h>
#include <pulsar/Result.h>
#include <log4cxx/logger.h>
#include <cpprest/json.h>

using namespace web;

using std::string;
using std::mutex;

namespace diffraflow {

    class AggSenderConsumer;

    class AggMetrics {
    public:
        explicit AggMetrics(string pulsar_url, int threads_count = 1);
        ~AggMetrics();

        void set_metrics(const string topic, const string key, const json::value& value);
        string get_metrics();

        bool start_sender_consumer(const string topic, int timeoutMs = 5000);
        void stop_sender_consumer();

    private:
        pulsar::Client* pulsar_client_;

        json::value metrics_json_;
        mutex metrics_json_mtx_;

        AggSenderConsumer* sender_consumer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif