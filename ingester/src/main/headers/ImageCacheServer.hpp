#ifndef ImageCacheServer_H
#define ImageCacheServer_H

#include <thread>
#include <vector>
#include <list>
#include <atomic>
#include <mutex>
#include <condition_variable>

using std::thread;
using std::vector;
using std::list;
using std::pair;
using std::atomic;
using std::mutex;
using std::condition_variable;

namespace shine {

    class ImageConnection;

    typedef list< pair<ImageConnection*, thread*> > connList;

    class ImageCacheServer {
    private:
        int port_;
        int server_sock_fd_;
        atomic<bool> server_run_;
        connList connections_;

        mutex mtx_;
        condition_variable cv_clean_;
        int clean_wait_time_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;

    public:
        ImageCacheServer(int p);
        ~ImageCacheServer();

        bool create_sock();
        int accept_client();

        void serve();

        void clean();

    };
}

#endif