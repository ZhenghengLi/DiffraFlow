#ifndef ImageData_H
#define ImageData_H

#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include "ObjectSerializer.hh"

namespace shine {

    class ImageFrame;

    class ImageData: public ObjectSerializer, private Decoder, private PrimitiveSerializer {
    private:
        size_t imgFrm_len_;
        ImageFrame* imgFrm_arr_;

    public:
        ImageData();
        ~ImageData();
        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);

        size_t serialize(char* const data, size_t len);
        size_t deserialize(const char* const data, size_t len);
        size_t object_size();
        int object_type();

    };
}

#endif