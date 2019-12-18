#ifndef ServerManager_H
#define ServerManager_H

#include <vector>
#include <algorithm>
#include <string>

using std::pair;
using std::string;
using std::vector;

namespace diffraflow {
    class ServerManager {
    public:
        ServerManager();
        ~ServerManager();

    private:
        // vector< pair<string, int> > read_address_list_(const char* filename);
        bool read_address_list_(const char* filename);
    };
}

#endif