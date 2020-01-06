#ifndef ImageData_H
#define ImageData_H

#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include "ObjectSerializer.hh"

namespace diffraflow {

    class ImageFrame;

    class ImageData: public ObjectSerializer {
    private:
        static const int obj_type_ = 1232;

    public:
        uint32_t     imgFrm_len;  // number of sub-detectors
        uint8_t*     status_arr;  // alignment status
        ImageFrame*  imgFrm_arr;  // image data from each sub-detector

        int64_t      event_key;   // equal to image_key
        double       event_time;  // equal to image_time

    private:
        void copyObj_(const ImageData& img_data);

    public:
        explicit ImageData(uint32_t numOfDet = 1);
        ImageData(const ImageData& img_data);
        ~ImageData();

        ImageData& operator=(const ImageData& img_data);

        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);

        void print();

        // serialization functions
        size_t serialize(char* const data, size_t len) override;
        size_t deserialize(const char* const data, size_t len) override;
        size_t object_size() override;
        int object_type() override;
        void clear_data() override;

    };
}

#endif
