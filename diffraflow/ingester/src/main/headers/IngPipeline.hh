#ifndef __IngPipeline_H__
#define __IngPipeline_H__

#include <atomic>
#include <log4cxx/logger.h>

using std::atomic_bool;

namespace diffraflow {

    class IngConfig;
    class IngImgWthFtrQueue;
    class IngImgDatFetcher;
    class IngCalibrationWorker;

    class IngPipeline {
    public:
        explicit IngPipeline(IngConfig* config);
        ~IngPipeline();

        void start_run();
        void terminate();

    private:
        IngConfig* config_obj_;

        IngImgDatFetcher* image_data_fetcher_;
        IngImgWthFtrQueue* imgWthFtrQue_raw_;

        // Calibration
        IngCalibrationWorker* calibration_worker_;
        IngImgWthFtrQueue* imgWthFtrQue_calib_;

        // Feature Extraction
        IngImgWthFtrQueue* imgWthFtrQue_feature_;

        // Filtering
        IngImgWthFtrQueue* imgWthFtrQue_write_;

        // Image Writer

        atomic_bool running_flag_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif