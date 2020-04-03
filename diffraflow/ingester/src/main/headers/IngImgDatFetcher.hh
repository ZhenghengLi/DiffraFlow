#ifndef __IngImgDatFetcher_H__
#define __IngImgDatFetcher_H__

#include "GenericClient.hh"
#include "IngImgDatRawQueue.hh"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <log4cxx/logger.h>

using std::mutex;
using std::condition_variable;
using std::atomic_bool;

namespace diffraflow {
    class IngImgDatFetcher: public GenericClient {
    public:
        IngImgDatFetcher(string combiner_host, int combiner_port,
            uint32_t ingester_id, IngImgDatRawQueue* raw_queue);
        ~IngImgDatFetcher();

        void set_recnxn_policy(size_t wait_time, size_t max_count);
        bool connect_to_combiner();

        bool start();

    private:
        size_t recnxn_wait_time_;
        size_t recnxn_max_count_;

        IngImgDatRawQueue* imgdat_raw_queue_;

        atomic_bool stopped_;
        mutex cnxn_mtx_;
        condition_variable cnxn_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif