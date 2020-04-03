#ifndef __IngImgDatFetcher_H__
#define __IngImgDatFetcher_H__

#include "GenericClient.hh"
#include "IngImgDatRawQueue.hh"

#include <log4cxx/logger.h>

namespace diffraflow {
    class IngImgDatFetcher: public GenericClient {
    public:
        IngImgDatFetcher(string combiner_host, int combiner_port,
            uint32_t ingester_id, IngImgDatRawQueue* raw_queue);
        ~IngImgDatFetcher();

        void set_recnxn_policy(int wait_time, int max_count);
        bool connect_to_combiner();

    private:
        int recnxn_wait_time_;
        int recnxn_max_count_;

        IngImgDatRawQueue* imgdat_raw_queue_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif