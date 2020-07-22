#ifndef __SndDatTran_H__
#define __SndDatTran_H__

#include <log4cxx/logger.h>
#include <fstream>
#include <mutex>

#include "GenericClient.hh"

using std::ifstream;
using std::string;
using std::mutex;

namespace diffraflow {

    class SndConfig;

    class SndDatTran : public GenericClient {
    public:
        explicit SndDatTran(SndConfig* conf_obj);
        ~SndDatTran();

        bool read_and_send(uint32_t event_index);

    private:
        SndConfig* config_obj_;
        char* head_buffer_;
        char* frame_buffer_;
        char* string_buffer_;

        int current_file_index_;
        ifstream* current_file_;
        string current_file_path_;
        mutex data_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif