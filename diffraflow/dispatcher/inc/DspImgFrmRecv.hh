#ifndef __DspImgFrmRecv_H__
#define __DspImgFrmRecv_H__

#include "GenericDgramReceiver.hh"
#include "BlockingQueue.hh"
#include <memory>
#include <thread>
#include <sched.h>

#define MOD_CNT 16

using std::shared_ptr;
using std::thread;

namespace diffraflow {

    class ImageFrameRaw;
    class DspSender;

    class DspImgFrmRecv : public GenericDgramReceiver {
    public:
        DspImgFrmRecv(string host, int port, DspSender** sender_arr, size_t sender_cnt,
            int rcvbufsize = 56 * 1024 * 1024, int dgram_queue_size = 16000);
        ~DspImgFrmRecv();

        void set_max_queue_size(size_t max_qs);

        void start_checker(cpu_set_t* cpuset = nullptr);
        void stop_checker();

    public:
        struct {
            atomic<uint64_t> total_received_count;
            atomic<uint64_t> total_checked_count;
        } frame_metrics;

        json::value collect_metrics() override;

    protected:
        void process_datagram_(shared_ptr<ByteBuffer>& datagram) override;

    private:
        uint32_t hash_long_(uint64_t value);

    private:
        shared_ptr<ImageFrameRaw> image_frame_arr_[MOD_CNT];
        DspSender** sender_array_;
        size_t sender_count_;

        BlockingQueue<shared_ptr<ImageFrameRaw>> imgfrm_queue_;
        thread* checker_thread_;

    private:
        static log4cxx::LoggerPtr logger_;
    };

} // namespace diffraflow

#endif