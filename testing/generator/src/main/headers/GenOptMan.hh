#ifndef GenOptMan_H
#define GenOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class GenOptMan : public OptionsManager {
    public:
        // option varables
        string data_dir;   // -d, --datdir=DIR
        int module_id;     // -i, --modid=ID
        string output_dir; // -o, --outdir=DIR
        string calib_file; // -c, --calib=FILE
        string align_file; // -a, --align=FILE
        string event_file; // -e, --event=FILE
        int max_events;    // -m, --maxevt=NUM

    public:
        GenOptMan();
        ~GenOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif