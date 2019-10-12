#ifndef ImageConnection_H
#define ImageConnection_H

#include <iostream>

namespace shine {
    class ImageConnection {
    private:
        char* buffer_;
        size_t buffer_size_;
    public:
        ImageConnection();
        ~ImageConnection();

    };
}

#endif