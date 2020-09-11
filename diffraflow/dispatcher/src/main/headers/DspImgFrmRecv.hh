#ifndef __DspImgFrmRecv_H__
#define __DspImgFrmRecv_H__

#include "GenericDgramReceiver.hh"
#include <memory>

#define MOD_CNT 16

using std::shared_ptr;

namespace diffraflow {

    class ImageFrameRaw;

    class DspImgFrmRecv : public GenericDgramReceiver {
    public:
        DspImgFrmRecv(string host, int port);
        ~DspImgFrmRecv();

    protected:
        void process_datagram_(shared_ptr<vector<char>>& datagram) override;

    private:
        shared_ptr<ImageFrameRaw> image_frame_arr[MOD_CNT];

    private:
        static log4cxx::LoggerPtr logger_;
    };

} // namespace diffraflow

#endif