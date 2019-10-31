#ifndef ImageData_H
#define ImageData_H

#include "Decoder.hh"

namespace shine {

    class ImageFrame;

    class ImageData: private Decoder {
    private:
        size_t imgFrm_len_;
        ImageFrame** imgFrm_arr_;

    public:
        ImageData();
        ~ImageData();
        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);
        bool serialize(char* data, size_t len);
        bool deserialize(char* data, size_t len);

    };
}

#endif