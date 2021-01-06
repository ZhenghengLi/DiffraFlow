#ifndef __IngPipeline_H__
#define __IngPipeline_H__

#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"
#include "IngImgWthFtrQueue.hh"

using std::atomic_bool;
using std::mutex;

namespace diffraflow {

    class IngConfig;
    class IngImgDatFetcher;
    class IngCalibrationWorker;
    class IngFeatureExtracter;
    class IngImageFilter;
    class IngImageWriter;
    class IngImgHttpServer;

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
        IngFeatureExtracter* feature_extracter_;
        IngImgWthFtrQueue* imgWthFtrQue_feature_;

        // Filtering
        IngImageFilter* image_filter_;
        IngImgWthFtrQueue* imgWthFtrQue_write_;

        // HTTP Server
        IngImgHttpServer* image_http_server_;

        // Image Writer
        IngImageWriter* image_writer_;

        atomic_bool running_flag_;
        mutex delete_mtx_;

        // metrics
        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif