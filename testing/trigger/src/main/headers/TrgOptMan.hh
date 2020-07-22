#ifndef TrgOptMan_H
#define TrgOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class TrgOptMan : public OptionsManager {
    public:
        // option varables
        string sender_list_file; // -s, --senderlist=FILE

    public:
        TrgOptMan();
        ~TrgOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];
    };

} // namespace diffraflow

#endif