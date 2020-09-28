#ifndef __AggBaseConsumer_H__
#define __AggBaseConsumer_H__

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
    class AggBaseConsumer {
    public:
        explicit AggBaseConsumer(string name);
        ~AggBaseConsumer();

        bool start(pulsar::Client* client, const string topic, int timeoutMs = 5000);
        void stop();

    public:
        enum Consumer_Status { kNotStart, kRunning, kStopped, kStopping };

        thread* consumer_thread_;
        atomic<Consumer_Status> consumer_status_;
        mutex consumer_mtx_;
        condition_variable consumer_cv_;

        mutex op_mtx_;

    protected:
        virtual void process_message(const pulsar::Message& message) = 0;

    private:
        string consumer_name_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif