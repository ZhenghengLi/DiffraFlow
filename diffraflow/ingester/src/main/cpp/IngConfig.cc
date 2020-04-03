#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <thread>

using std::cout;
using std::endl;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::IngConfig::logger_
    = log4cxx::Logger::getLogger("IngConfig");

diffraflow::IngConfig::IngConfig() {
    ingester_id = 0;
    combiner_port = -1;
    combiner_host = "localhost";
    recnxn_wait_time = 0;
    recnxn_max_count = 0;
    imgdat_queue_capacity = 100;

    zookeeper_setting_ready_flag_ = false;

    // initial values of dynamic configurations
    dy_param_int_ = 20;
    dy_param_double_ = 100;
    dy_param_string_ = "xfel";

}

diffraflow::IngConfig::~IngConfig() {

}

bool diffraflow::IngConfig::load(const char* filename) {
    list< pair<string, string> > conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    if (zookeeper_parse_setting_(conf_KV_list)) {
        zookeeper_setting_ready_flag_ = true;
    } else {
        LOG4CXX_WARN(logger_, "zookeeper setting is not ready, configuration will not be dynamically updated.")
    }
    map<string, string> dy_conf_map;
    for (list< pair<string, string> >::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        // for static parameters
        if (key == "ingester_id") {
            ingester_id = atoi(value.c_str());
        } else if (key == "combiner_host") {
            combiner_host = value;
        } else if (key == "combiner_port") {
            combiner_port = atoi(value.c_str());
        } else if (key == "recnxn_wait_time") {
            recnxn_wait_time = atoi(value.c_str());
        } else if (key == "recnxn_max_count") {
            recnxn_max_count = atoi(value.c_str());
        } else if (key == "imgdat_queue_capacity") {
            imgdat_queue_capacity = atoi(value.c_str());
        // for dynamic parameters
        } else {
            dy_conf_map[key] = value;
        }
    }
    // use POD IP as ingester_id
    if (ingester_id == 0) {
        const char* pod_ip = getenv("POD_IP");
        if (pod_ip != NULL) {
            vector<string> ip_nums;
            boost::split(ip_nums, pod_ip, boost::is_any_of("."));
            for (size_t i = 0; i < ip_nums.size(); i++) {
                ingester_id <<= 8;
                ingester_id += atoi(ip_nums[i].c_str());
            }
        }
    }
    // validation check for static parameters
    bool succ_flag = true;
    if (combiner_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid combiner_port: " << combiner_port);
        succ_flag = false;
    }
    if (imgdat_queue_capacity < 1 || imgdat_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "imgdat_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    // check and commit for dynamic parameters
    if (!check_and_commit_(dy_conf_map, time(NULL))) {
        LOG4CXX_ERROR(logger_, "dynamic configurations have invalid values.");
        succ_flag = false;
    }
    if (succ_flag) {
        ingester_config_json_["ingester_id"] = json::value::number(ingester_id);
        ingester_config_json_["combiner_host"] = json::value::string(combiner_host);
        ingester_config_json_["combiner_port"] = json::value::number(combiner_port);
        return true;
    } else {
        return false;
    }
}

json::value diffraflow::IngConfig::collect_metrics() {
    json::value root_json;
    if (zookeeper_setting_ready_flag_) {
        root_json = DynamicConfiguration::collect_metrics();
    }
    lock_guard<mutex> ingester_config_json_lg(ingester_config_json_mtx_);
    root_json["ingester_config"] = ingester_config_json_;
    return root_json;
}

bool diffraflow::IngConfig::zookeeper_setting_is_ready() {
    return zookeeper_setting_ready_flag_;
}

void diffraflow::IngConfig::print() {
    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    if (zookeeper_setting_ready_flag_) {
        zookeeper_print_setting();
    }

    cout << "server config:" << endl;
    cout << "- ingester_id = " << ingester_id << endl;
    cout << "- combiner_host = " << combiner_host << endl;
    cout << "- combiner_port = " << combiner_port << endl;
    cout << "- recnxn_wait_time = " << recnxn_wait_time << endl;
    cout << "- recnxn_max_count = " << recnxn_max_count << endl;
    cout << "dynamic parameters:" << endl;
    cout << "- dy_param_int = " << dy_param_int_.load() << endl;
    cout << "- dy_param_double = " << dy_param_double_.load() << endl;
    cout << "- dy_param_string = " << dy_param_string_ << endl;

}

bool diffraflow::IngConfig::check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime) {

    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    // values before commit
    int tmp_dy_param_int = dy_param_int_.load();
    double tmp_dy_param_double = dy_param_double_.load();
    string tmp_dy_param_string = dy_param_string_;

    // convert
    for (map<string, string>::const_iterator iter = conf_map.begin(); iter != conf_map.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "dy_param_int") {
            tmp_dy_param_int = atoi(value.c_str());
        } else if (key == "dy_param_double") {
            tmp_dy_param_double = atof(value.c_str());
        } else if (key == "dy_param_string") {
            tmp_dy_param_string = value;
        }
    }

    // validation check
    bool invalid_flag = false;
    if (tmp_dy_param_int < 10) {
        cout << "invalid configuration: dy_param_int(" << tmp_dy_param_int << ") is out of range [10, inf)." << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_double > 1000) {
        cout << "invalid configuration: dy_param_double(" << tmp_dy_param_double << ") is out of range (-inf, 1000]." << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_string.length() < 2) {
        cout << "invalid configuration: dy_param_string(" << tmp_dy_param_string << ") is too short." << endl;
        invalid_flag = true;
    }

    if (invalid_flag) {
        return false;
    }

    // commit change
    if (dy_param_int_ != tmp_dy_param_int) {
        cout << "configuration changed: dy_param_int [ " << dy_param_int_ << " -> " << tmp_dy_param_int << " ]." << endl;
        dy_param_int_ = tmp_dy_param_int;
    }
    if (dy_param_double_ != tmp_dy_param_double) {
        cout << "configuration changed: dy_param_double [ " << dy_param_double_ << " -> " << tmp_dy_param_double << " ]." << endl;
        dy_param_double_ = tmp_dy_param_double;
    }
    if (dy_param_string_ != tmp_dy_param_string) {
        cout << "configuration changed: dy_param_string [ " << dy_param_string_ << " -> " << tmp_dy_param_string << " ]." << endl;
        dy_param_string_ = tmp_dy_param_string;
    }

    config_mtime_ = conf_mtime;

    lock_guard<mutex> ingester_config_json_lg(ingester_config_json_mtx_);
    ingester_config_json_["dy_param_int"] = json::value::number(dy_param_int_);
    ingester_config_json_["dy_param_double"] = json::value::number(dy_param_double_);
    ingester_config_json_["dy_param_string"] = json::value::string(dy_param_string_);
    ingester_config_json_["config_mtime"] = json::value::string(boost::trim_copy(string(ctime(&config_mtime_))));

    return true;
}

int diffraflow::IngConfig::get_dy_param_int() {
    return dy_param_int_.load();
}

double diffraflow::IngConfig::get_dy_param_double() {
    return dy_param_double_.load();
}

string diffraflow::IngConfig::get_dy_param_string() {
    lock_guard<mutex> lg(dy_param_string_mtx_);
    return dy_param_string_;
}
