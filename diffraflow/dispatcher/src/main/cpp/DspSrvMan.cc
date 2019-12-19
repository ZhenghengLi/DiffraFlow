#include "DspSrvMan.hh"

#include <fstream>

#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;

diffraflow::DspSrvMan::DspSrvMan() {

}

diffraflow::DspSrvMan::~DspSrvMan() {

}

bool diffraflow::DspSrvMan::read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec) {
    addr_vec.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        BOOST_LOG_TRIVIAL(error) << "address file open failed.";
        return false;
    }
    string oneline;
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        // extract host and port
        vector<string> host_port;
        boost::split(host_port, oneline, boost::is_any_of(":"));
        if (host_port.size() < 2) {
            BOOST_LOG_TRIVIAL(error) << "found unknown address: " << oneline;
            return false;
        }
        boost::trim(host_port[0]);
        string host_str = host_port[0];
        boost::trim(host_port[1]);
        int    port_num = atoi(host_port[1].c_str());
        if (port_num <= 0) {
            BOOST_LOG_TRIVIAL(error) << "found unknown address: " << oneline;
            return false;
        }
        addr_vec.push_back(make_pair(host_str, port_num));
    }
    if (addr_vec.size() > 0) {
        return true;
    } else {
        return false;
    }
}
