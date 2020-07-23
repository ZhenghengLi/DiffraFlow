#ifndef __TrgCoordinator_H__
#define __TrgCoordinator_H__

#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <atomic>
#include <condition_variable>

#include <log4cxx/logger.h>

using std::vector;
using std::pair;
using std::string;
using std::atomic_int;
using std::atomic_bool;
using std::atomic;
using std::mutex;
using std::condition_variable;
using std::thread;

namespace diffraflow {

    class TrgClient;

    class TrgCoordinator {
    public:
        TrgCoordinator();
        ~TrgCoordinator();

        bool create_trigger_clients(const char* sender_list_file, uint32_t trigger_id);
        void delete_trigger_clients();

        bool trigger_one_event(uint32_t event_index);
        bool trigger_many_events(uint32_t start_event_index, uint32_t event_count, uint32_t interval_microseconds);

    private:
        bool read_address_list_(const char* filename, vector<pair<string, int>>& addr_vec);

    private:
        TrgClient** trigger_client_arr_;
        size_t trigger_client_cnt_;

        atomic_bool running_flag_;

        atomic_bool* trigger_flag_arr_;
        mutex* trigger_mtx_arr_;
        condition_variable* trigger_cv_arr_;
        thread** trigger_thread_arr_;

        atomic<uint32_t> event_index_to_trigger_;
        atomic_int count_down_;
        mutex event_mtx_;
        condition_variable event_cv_;

        vector<int> fail_mod_ids_vec_;
        mutex fail_mod_ids_mtx_;

        mutex wait_mtx_;
        condition_variable wait_cv_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

} // namespace diffraflow
#endif