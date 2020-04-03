#ifndef __IngPipeline_H__
#define __IngPipeline_H__

#include <atomic>
#include <log4cxx/logger.h>

using std::atomic_bool;

namespace diffraflow {

    class IngConfig;
    class IngImgDatRawQueue;
    class IngImgDatFetcher;

    class IngPipeline {
    public:
        IngPipeline(IngConfig* config);
        ~IngPipeline();

        void start_run();
        void terminate();

    private:
        IngConfig* config_obj_;

        IngImgDatRawQueue* image_data_raw_queue_;
        IngImgDatFetcher* image_data_fetcher_;

        atomic_bool running_flag_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif