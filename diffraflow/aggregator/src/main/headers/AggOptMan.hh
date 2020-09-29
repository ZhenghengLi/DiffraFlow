#ifndef __AggOptMan_H__
#define __AggOptMan_H__

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class AggOptMan : public OptionsManager {
    public:
        // option varables
        string config_file;  // -c, --config=FILE
        string logconf_file; // -l, --logconf=FILE

    public:
        AggOptMan();
        ~AggOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif