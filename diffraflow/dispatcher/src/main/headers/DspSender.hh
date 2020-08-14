#ifndef DspSender_H
#define DspSender_H

#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <atomic>
#include <log4cxx/logger.h>

#include "GenericClient.hh"

using std::string;
using std::thread;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;
using std::atomic;
using std::atomic_bool;

namespace diffraflow {
    class DspSender : public GenericClient {
    public:
        DspSender(string hostname, int port, int id);
        ~DspSender();

        bool send(const char* data, const size_t len);

    private:
        mutex mtx_send_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
