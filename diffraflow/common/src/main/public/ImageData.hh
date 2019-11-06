#ifndef ImageData_H
#define ImageData_H

#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include "ObjectSerializer.hh"

namespace shine {

    class ImageFrame;

    class ImageData: public ObjectSerializer {
    public:
        uint32_t     imgFrm_len;  // number of sub-detectors
        ImageFrame*  imgFrm_arr;  // image data from each sub-detector
        uint8_t*     status_arr;  // alignment status
        
        int64_t      event_key;   // equal to image_key
        double       event_time;  // equal to image_time

    private:
        void copyObj_(const ImageData& img_data);

    public:
        ImageData();
        ImageData(uint32_t numOfDet);
        ImageData(const ImageData& img_data);
        ~ImageData();

        ImageData& operator=(const ImageData& img_data);

        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);

        size_t serialize(char* const data, size_t len);
        size_t deserialize(const char* const data, size_t len);
        size_t object_size();
        int object_type();
        void clear_data();

    };
}

#endif