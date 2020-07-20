#ifndef SndOptMan_H
#define SndOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class SndOptMan : public OptionsManager {
    public:
        // option varables
        string config_file;  // -c, --config=FILE
        string logconf_file; // -l, --logconf=FILE
        string nodemap_file; // -n, --nodemap=FILE

    public:
        SndOptMan();
        ~SndOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif