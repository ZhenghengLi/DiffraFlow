#ifndef GenOptMan_H
#define GenOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class FtrOptMan : public OptionsManager {
    public:
        // option varables
        string data_file;    // -i, --datfile=FILE
        string output_file;  // -o, --outfile=FILE
        string config_file;  // -c, --config=FILE
        string logconf_file; // -l, --logconf=FILE

    public:
        FtrOptMan();
        ~FtrOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif