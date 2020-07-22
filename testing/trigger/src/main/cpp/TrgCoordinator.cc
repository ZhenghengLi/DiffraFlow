#include "TrgCoordinator.hh"

#include <fstream>
#include <vector>

using std::ifstream;
using std::vector;
using std::string;
using std::pair;

diffraflow::TrgCoordinator::TrgCoordinator() {
    trigger_client_arr_ = nullptr;
    trigger_client_cnt_ = 0;
}

diffraflow::TrgCoordinator::~TrgCoordinator() {
    if (trigger_client_arr_ != nullptr) {
        delete[] trigger_client_arr_;
    }
}

bool diffraflow::TrgCoordinator::create_trigger_clients(const char* sender_list_file) {
    // read sender list
    vector<pair<string, int>> host_port_vec;
    ifstream infile;

    return true;
}

bool diffraflow::TrgCoordinator::send_one_event(uint32_t event_index) {
    // send

    return true;
}

bool diffraflow::TrgCoordinator::send_many_events(
    uint32_t start_event_index, uint32_t event_count, uint32_t interval_ms) {
    // send

    return true;
}