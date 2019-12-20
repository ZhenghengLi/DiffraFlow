#ifndef DspSrvMan_H
#define DspSrvMan_H

#include <vector>
#include <algorithm>
#include <string>

using std::pair;
using std::string;
using std::vector;

namespace diffraflow {

    class DspConfig;
    class DspSender;

    class DspSrvMan {
    public:
        DspSrvMan(DspConfig* config);
        ~DspSrvMan();

        bool create_senders(const char* address_list_fn);
        void delete_senders();

    private:
        bool read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec);

    private:
        DspConfig*  config_obj_;
        DspSender** sender_arr_;
        size_t      sender_cnt_;

    };
}

#endif
