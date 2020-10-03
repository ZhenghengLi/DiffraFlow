#ifndef __AggMetrics_H__
#define __AggMetrics_H__

#include <string>
#include <map>
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
using std::map;
using std::condition_variable;

namespace diffraflow {

    class AggConsumer;

    class AggMetrics {
        friend class AggConsumer;

    public:
        explicit AggMetrics(string pulsar_url, string sub_name = "aggregator", bool rd_compact = false,
            int io_threads_count = 1, int listener_threads_count = 1);
        ~AggMetrics();

        json::value get_metrics();

        bool start_consumer(const string name, const string topic, int timeoutMs = 5000);
        void stop_consumer(const string name);

        void wait();
        void stop_all();

    private:
        void set_metrics_(const string topic, const string key, const json::value& value);

    private:
        pulsar::Client* pulsar_client_;
        string subscription_name_;
        bool read_compacted_;

        json::value metrics_json_;
        mutex metrics_json_mtx_;

        map<string, AggConsumer*> consumer_map_;
        mutex consumer_map_mtx_;
        condition_variable consumer_map_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif