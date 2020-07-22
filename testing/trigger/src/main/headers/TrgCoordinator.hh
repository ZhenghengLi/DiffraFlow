#ifndef __TrgCoordinator_H__
#define __TrgCoordinator_H__

#include <thread>
#include <mutex>

namespace diffraflow {

    class TrgClient;

    class TrgCoordinator {
    public:
        TrgCoordinator();
        ~TrgCoordinator();

        bool create_trigger_clients(const char* sender_list_file);

        bool send_one_event(uint32_t event_index);
        bool send_many_events(uint32_t start_event_index, uint32_t event_count, uint32_t interval_ms);

    private:
        TrgClient* trigger_client_arr_;
        size_t trigger_client_cnt_;
    };

} // namespace diffraflow
#endif