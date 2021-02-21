#ifndef DspOptMan_H
#define DspOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class DspOptMan : public OptionsManager {
    public:
        // option varables
        string config_file;  // -c, --config=FILE
        string logconf_file; // -l, --logconf=FILE
        string cmbaddr_file; // -a, --cmbaddr=FILE

    public:
        DspOptMan();
        ~DspOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif