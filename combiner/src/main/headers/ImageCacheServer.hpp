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
    class ImageCache;

    typedef list< pair<ImageConnection*, thread*> > connList;

    class ImageCacheServer {
    private:
        int server_sock_fd_;
        atomic<bool> server_run_;
        connList connections_;
        int read_timeout_;

        mutex mtx_;
        condition_variable cv_clean_;
        int clean_wait_time_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;

        ImageCache* image_cache_;

    private:
        bool create_sock_(int port);
        int accept_client_();
        void clean_();

    public:
        ImageCacheServer();
        ~ImageCacheServer();

        void serve(int port);
        void stop();

        ImageCache* image_cache() {
            return image_cache_;
        }

    };
}

#endif