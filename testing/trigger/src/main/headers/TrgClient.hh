#ifndef __TrgClient_H__
#define __TrgClient_H__

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <log4cxx/logger.h>

#include "GenericClient.hh"

using std::thread;
using std::mutex;
using std::condition_variable;
using std::atomic_bool;

namespace diffraflow {
    class TrgClient : public GenericClient {
    public:
        TrgClient(string sender_host, int sender_port, uint32_t trigger_id);
        ~TrgClient();

        bool start();
        void stop();

        bool trigger(const uint32_t event_index);
        bool wait();

    private:
        bool send_event_index_(const uint32_t event_index);

    private:
        char* send_buffer_;
        char* recv_buffer_;

        thread* worker_;
        mutex trigger_mtx_;
        condition_variable trigger_cv_;
        condition_variable wait_cv_;
        bool trigger_flag_;
        bool running_flag_;
        uint32_t target_event_index_;
        bool success_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif