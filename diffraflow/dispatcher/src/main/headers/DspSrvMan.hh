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
        bool read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec);
    };
}

#endif
