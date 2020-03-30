#ifndef GenericServer_H
#define GenericServer_H

#include <thread>
#include <vector>
#include <list>
#include <atomic>
#include <mutex>
#include <future>
#include <condition_variable>
#include <log4cxx/logger.h>

#include "MetricsProvider.hh"

using std::string;
using std::thread;
using std::vector;
using std::list;
using std::pair;
using std::atomic;
using std::mutex;
using std::condition_variable;
using std::future;
using std::async;

namespace diffraflow {

    class GenericConnection;

    class GenericServer: public MetricsProvider {
    public:
        explicit GenericServer(string host, int port, size_t max_conn = 100);
        explicit GenericServer(string sock_path, size_t max_conn = 100);
        virtual ~GenericServer();

        void start();
        void stop();
        void wait();
        int  get();

    private:
        int serve_();
        future<int> worker_;

    public:
        virtual json::value collect_metrics() override;

    protected:
        // methods to be implemented
        virtual GenericConnection* new_connection_(int client_sock_fd) = 0;

    protected:
        bool create_tcp_sock_();
        bool create_ipc_sock_();
        int accept_client_();
        void clean_();
        void start_cleaner_();
        void stop_cleaner_();

    protected:
        typedef list< pair<GenericConnection*, thread*> > connListT_;

    protected:
        bool is_ipc_;
        string server_sock_host_;
        int    server_sock_port_;
        string server_sock_path_;
        int server_sock_fd_;
        atomic<bool> server_run_;
        connListT_ connections_;
        size_t max_conn_counts_;

        mutex mtx_conn_;
        condition_variable cv_clean_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;
        atomic<int> dead_counts_;

        atomic<bool> server_finished_;
        mutex mtx_finish_;
        condition_variable cv_finish_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
