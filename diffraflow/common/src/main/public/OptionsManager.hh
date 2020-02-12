#ifndef OptionsManager_H
#define OptionsManager_H

#include <string>

using std::string;

namespace diffraflow {
    class OptionsManager {
    public:
        explicit OptionsManager(const char* sw_name, const char* sw_description);
        virtual ~OptionsManager();

        virtual bool parse(int argc, char** argv) = 0;
        virtual void print();

    protected:
        virtual void print_help_() = 0;
        virtual void print_version_();

    protected:
        string software_name_;
        string software_description_;
        bool version_flag_;

    };
}

#endif