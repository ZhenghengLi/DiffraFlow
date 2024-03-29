#ifndef __IngPipeline_H__
#define __IngPipeline_H__

#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"
#include "IngBufferItemQueue.hh"

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
    class IngImgFtrBuffer;

    class IngPipeline {
    public:
        explicit IngPipeline(IngConfig* config);
        ~IngPipeline();

        void start_run();
        void terminate();

    private:
        IngConfig* config_obj_;
        IngImgFtrBuffer* image_feature_buffer_;

        IngImgDatFetcher* image_data_fetcher_;
        IngBufferItemQueue* item_queue_raw_;

        // Calibration
        IngCalibrationWorker* calibration_worker_;
        IngBufferItemQueue* item_queue_calib_;

        // Feature Extraction
        IngFeatureExtracter* feature_extracter_;
        IngBufferItemQueue* item_queue_feature_;

        // Filtering
        IngImageFilter* image_filter_;
        IngBufferItemQueue* item_queue_write_;

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