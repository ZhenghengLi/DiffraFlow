#ifndef __CtrOptMan_H__
#define __CtrOptMan_H__

#include "OptionsManager.hh"
#include <getopt.h>

#include <vector>

using std::vector;

namespace diffraflow {
    class CtrOptMan: public OptionsManager {
    public:
        // option varables
        string config_file;             // -c, --config=FILE
        string zk_conf_file;            // -z, --zkconfig=FILE
        string logconf_file;            // -l, --logconf=FILE
        string monaddr_file;            // -a, --monaddr=FILE

        vector<string> zk_actions;      // -C, --zkcreate=ZNODE:FILE
                                        // -D, --zkdelete=ZNODE
                                        // -U, --zkupdate=ZNODE:FILE
                                        // -R, --zkread=ZNODE
                                        // -L, --zklist=ZNODE

    public:
        CtrOptMan();
        ~CtrOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];


    };
}

#endif