#ifndef __AggConsumer_H__
#define __AggConsumer_H__

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <pulsar/Client.h>
#include <pulsar/Consumer.h>
#include <pulsar/Message.h>
#include <pulsar/Result.h>
#include <log4cxx/logger.h>

using std::string;
using std::thread;
using std::mutex;
using std::atomic;
using std::condition_variable;

namespace diffraflow {

    class AggMetrics;

    class AggConsumer {
    public:
        explicit AggConsumer(AggMetrics* metrics, const string name, const string topic);
        virtual ~AggConsumer();

        bool start(int timeoutMs = 5000);
        void stopping();
        void stop();
        void wait();

    private:
        void process_message_(const pulsar::Message& message);

    private:
        enum Consumer_Status_ { kNotStart, kRunning, kStopped, kStopping };

        thread* consumer_thread_;
        atomic<Consumer_Status_> consumer_status_;
        mutex consumer_mtx_;
        condition_variable consumer_cv_;

        mutex op_mtx_;

        AggMetrics* aggregated_metrics_;
        string consumer_name_;
        string consumer_topic_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif