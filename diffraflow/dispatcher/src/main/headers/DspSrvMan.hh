#ifndef DspSrvMan_H
#define DspSrvMan_H

#include <vector>
#include <algorithm>
#include <string>
#include <atomic>

using std::pair;
using std::string;
using std::vector;
using std::atomic_bool;

namespace diffraflow {

    class DspConfig;
    class DspSender;
    class DspImgFrmSrv;

    class DspSrvMan {
    public:
        explicit DspSrvMan(DspConfig* config);
        ~DspSrvMan();

        void start_run();
        void terminate();

    private:
        bool create_senders_(const char* address_list_fn, int dispatcher_id, bool compress_flag = false);
        void delete_senders_();
        bool read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec);

    private:
        DspConfig*  config_obj_;
        DspSender** sender_arr_;
        size_t      sender_cnt_;

        DspImgFrmSrv* imgfrm_srv_;
        atomic_bool running_flag_;

    };
}

#endif
