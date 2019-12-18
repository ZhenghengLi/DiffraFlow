#ifndef DispatchingServer_H
#define DispatchingServer_H

#include "GeneralServer.hh"

#include <cstddef>
#include <vector>
#include <string>

using std::vector;
using std::string;

namespace diffraflow {

    class Sender;

    class DispatchingServer: public GeneralServer {
    public:
        DispatchingServer(int port, Sender** sender_arr, size_t sender_cnt);
        ~DispatchingServer();

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    private:
        Sender** sender_array_;
        size_t   sender_count_;
    };
}

#endif