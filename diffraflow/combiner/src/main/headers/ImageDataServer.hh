#ifndef ImageDataServer_H
#define ImageDataServer_H

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

    class ImageDataServer {
    private:
        ImageCache* image_cache_;

    public:
        ImageDataServer(ImageCache* img_cache);
        ~ImageDataServer();

    };
}

#endif