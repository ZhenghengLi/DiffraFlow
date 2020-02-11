#ifndef GenericServer_H
#define GenericServer_H

#include <thread>
#include <vector>
#include <list>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::string;
using std::thread;
using std::vector;
using std::list;
using std::pair;
using std::atomic;
using std::mutex;
using std::condition_variable;

namespace diffraflow {

    class GenericConnection;

    class GenericServer {
    private:
        typedef list< pair<GenericConnection*, thread*> > connListT_;

    private:
        bool is_ipc_;
        string server_sock_host_;
        int    server_sock_port_;
        string server_sock_path_;
        int server_sock_fd_;
        atomic<bool> server_run_;
        connListT_ connections_;

        mutex mtx_;
        condition_variable cv_clean_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;
        atomic<int> dead_counts_;

        log4cxx::LoggerPtr logger_;

    private:
        bool create_tcp_sock_();
        bool create_ipc_sock_();
        int accept_client_();
        void clean_();
        void start_cleaner_();
        void stop_cleaner_();

    protected:
        // methods to be implemented
        virtual GenericConnection* new_connection_(int client_sock_fd) = 0;

    public:
        explicit GenericServer(string host, int port);
        explicit GenericServer(string sock_path);
        virtual ~GenericServer();

        void serve();
        void stop();

    };
}

#endif
