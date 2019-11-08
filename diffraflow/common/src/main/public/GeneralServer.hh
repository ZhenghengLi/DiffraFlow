#ifndef GeneralServer_H
#define GeneralServer_H

#include <thread>
#include <vector>
#include <list>
#include <atomic>
#include <mutex>
#include <condition_variable>

using std::string;
using std::thread;
using std::vector;
using std::list;
using std::pair;
using std::atomic;
using std::mutex;
using std::condition_variable;

namespace shine {

    class GeneralConnection;

    class GeneralServer {
    private:
        typedef list< pair<GeneralConnection*, thread*> > connListT_;

    private:
        bool is_ipc_;
        int server_sock_port_;
        string server_sock_path_;
        int server_sock_fd_;
        atomic<bool> server_run_;
        connListT_ connections_;

        mutex mtx_;
        condition_variable cv_clean_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;
        atomic<int> dead_counts_;

    private:
        bool create_tcp_sock_();
        bool create_ipc_sock_();
        int accept_client_();
        void clean_();
        void start_cleaner_();

    protected:
        // methods to be implemented
        virtual GeneralConnection* new_connection_(int client_sock_fd) = 0;

    public:
        GeneralServer(int port);
        GeneralServer(string sock_path);
        ~GeneralServer();

        void serve();
        void stop();

    };
}

#endif