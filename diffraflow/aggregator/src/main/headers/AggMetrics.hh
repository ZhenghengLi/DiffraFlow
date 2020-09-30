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
    class AggDispatcherConsumer;
    class AggCombinerConsumer;
    class AggIngesterConsumer;
    class AggMonitorConsumer;

    class AggMetrics {
    public:
        explicit AggMetrics(string pulsar_url, int threads_count = 1);
        ~AggMetrics();

        void set_metrics(const string topic, const string key, const json::value& value);
        json::value get_metrics();

        bool start_sender_consumer(const string topic, int timeoutMs = 5000);
        void stopping_sender_consumer();
        void stop_sender_consumer();

        bool start_dispatcher_consumer(const string topic, int timeoutMs = 5000);
        void stopping_dispatcher_consumer();
        void stop_dispatcher_consumer();

        bool start_combiner_consumer(const string topic, int timeoutMs = 5000);
        void stopping_combiner_consumer();
        void stop_combiner_consumer();

        bool start_ingester_consumer(const string topic, int timeoutMs = 5000);
        void stopping_ingester_consumer();
        void stop_ingester_consumer();

        bool start_monitor_consumer(const string topic, int timeoutMs = 5000);
        void stopping_monitor_consumer();
        void stop_monitor_consumer();

        void wait_all();
        void stop_all();

    private:
        pulsar::Client* pulsar_client_;

        json::value metrics_json_;
        mutex metrics_json_mtx_;

        AggSenderConsumer* sender_consumer_;
        AggDispatcherConsumer* dispatcher_consumer_;
        AggCombinerConsumer* combiner_consumer_;
        AggIngesterConsumer* ingester_consumer_;
        AggMonitorConsumer* monitor_consumer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif