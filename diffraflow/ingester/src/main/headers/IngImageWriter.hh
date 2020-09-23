#ifndef __IngImageWriter_H__
#define __IngImageWriter_H__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <log4cxx/logger.h>
#include "ImageFileHDF5W.hh"
#include "ImageFileRawW.hh"
#include "MetricsProvider.hh"

using std::mutex;
using std::condition_variable;
using std::atomic_bool;
using std::atomic;
using std::future;
using std::shared_future;
using std::shared_ptr;
using std::async;
using std::string;

namespace diffraflow {

    struct ImageWithFeature;

    class IngImgWthFtrQueue;
    class ImageFeature;
    class IngConfig;

    class IngImageWriter : public MetricsProvider {
    public:
        IngImageWriter(IngImgWthFtrQueue* img_queue_in, IngConfig* conf_obj);

        ~IngImageWriter();

        bool start();
        void wait();
        int stop();

    public:
        json::value collect_metrics() override;

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        int run_();
        shared_future<int> worker_;

        atomic<WorkerStatus> worker_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        bool create_directories_();
        bool open_file_();
        void close_file_();

        bool save_image_(const shared_ptr<ImageWithFeature>& image_with_feature);

    private:
        IngImgWthFtrQueue* image_queue_in_;
        IngConfig* config_obj_;

        // files:
        ImageFileHDF5W* image_file_hdf5_;
        ImageFileRawW* image_file_raw_;

        // file path:
        // storage_dir/R0000/NODENAME_N00/T00/R0000_NODENAME_N00_T00_S0000.h5
        atomic<int> current_run_number_;
        atomic<int> current_turn_number_;
        atomic<int> current_sequence_number_;
        atomic<int> current_saved_counts_;
        atomic<uint64_t> total_saved_counts_;
        atomic<int> total_opened_counts_;
        string current_folder_path_string_;
        string current_filename_string_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif