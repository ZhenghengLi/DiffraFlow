#ifndef ImageFrameServer_H
#define ImageFrameServer_H

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

    class ImageFrameConnection;
    class ImageCache;

    class ImageFrameServer {
    private:
        typedef list< pair<ImageFrameConnection*, thread*> > connListT_;

    private:
        int server_sock_fd_;
        atomic<bool> server_run_;
        connListT_ connections_;

        mutex mtx_;
        condition_variable cv_clean_;
        thread* cleaner_;
        atomic<bool> cleaner_run_;
        atomic<int> dead_counts_;

        ImageCache* image_cache_;

    private:
        bool create_sock_(int port);
        int accept_client_();
        void clean_();

    public:
        ImageFrameServer(ImageCache* img_cache);
        ~ImageFrameServer();

        void serve(int port);
        void stop();

    };
}

#endif