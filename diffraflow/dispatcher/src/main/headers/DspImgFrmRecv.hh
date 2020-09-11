#ifndef __DspImgFrmRecv_H__
#define __DspImgFrmRecv_H__

#include "GenericDgramReceiver.hh"

namespace diffraflow {
    class DspImgFrmRecv : public GenericDgramReceiver {
    public:
        DspImgFrmRecv(string host, int port);
        ~DspImgFrmRecv();

    protected:
        void process_datagram_(shared_ptr<vector<char>>& datagram) override;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif