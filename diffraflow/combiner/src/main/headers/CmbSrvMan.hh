#ifndef CmbSrvMan_H
#define CmbSrvMan_H

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
        CmbImgFrmSrv* image_frame_server_;

    };
}

#endif
