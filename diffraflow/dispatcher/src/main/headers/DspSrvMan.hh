#ifndef DspSrvMan_H
#define DspSrvMan_H

#include <vector>
#include <algorithm>
#include <string>
#include <atomic>
#include <log4cxx/logger.h>

#include "DspSender.hh"
#include "MetricsReporter.hh"

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
        DspSrvMan(DspConfig* config, const char* cmbaddr_file);
        ~DspSrvMan();

        void start_run();
        void terminate();

    private:
        bool create_senders_(const char* address_list_fn, int dispatcher_id,
            DspSender::CompressMethod compress_method = DspSender::kNone, int compress_level = 1);
        void delete_senders_();
        bool read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec);

    private:
        DspConfig*  config_obj_;
        DspSender** sender_arr_;
        size_t      sender_cnt_;

        DspImgFrmSrv* imgfrm_srv_;
        atomic_bool running_flag_;

        string combiner_address_file_;

        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
