#ifndef __IngImgRestServer_H__
#define __IngImgRestServer_H__

#include <string>
#include <cpprest/http_listener.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

using std::string;
using web::http::experimental::listener::http_listener;
using web::http::http_request;

namespace diffraflow {

    class IngImageFilter;

    class IngImgRestServer {
    public:
        explicit IngImgRestServer(IngImageFilter* img_filter);
        ~IngImgRestServer();

        bool start(string host, int port);
        void stop();

    private:
        void handleGet_(http_request message);

    private:
        IngImageFilter* image_filter_;
        http_listener*  listener_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif