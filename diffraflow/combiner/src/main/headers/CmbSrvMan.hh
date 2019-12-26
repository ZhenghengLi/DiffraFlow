#ifndef CmbSrvMan_H
#define CmbSrvMan_H

#include <atomic>

using std::atomic_bool;

namespace diffraflow {

    class CmbConfig;
    class CmbImgCache;
    class CmbImgFrmSrv;

    class CmbSrvMan {
    public:
        explicit CmbSrvMan(CmbConfig* config);
        ~CmbSrvMan();

        void start_run();
        void terminate();

    private:
        CmbConfig* config_obj_;
        CmbImgCache* image_cache_;

        CmbImgFrmSrv* imgfrm_srv_;
        atomic_bool running_flag_;

    };
}

#endif
