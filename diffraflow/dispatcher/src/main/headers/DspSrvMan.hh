#ifndef DspSrvMan_H
#define DspSrvMan_H

#include <vector>
#include <algorithm>
#include <string>

using std::pair;
using std::string;
using std::vector;

namespace diffraflow {
    class DspSrvMan {
    public:
        DspSrvMan();
        ~DspSrvMan();

    private:
        // vector< pair<string, int> > read_address_list_(const char* filename);
        bool read_address_list_(const char* filename);
    };
}

#endif
