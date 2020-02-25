#ifndef __CtrOptMan_H__
#define __CtrOptMan_H__

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class CtrOptMan: public OptionsManager {
    public:
        // option varables
        string config_file;       // -c, --config=FILE
        string zk_conf_file;      // -z, --zkconfig=FILE
        string logconf_file;      // -l, --logconf=FILE
        string zk_create;         // -C, --zkcreate=ZNODE:FILE
        string zk_delete;         // -D, --zkdelete=ZNODE
        string zk_update;         // -U, --zkupdate=ZNODE:FILE
        string zk_read;           // -R, --zkread=ZNODE

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