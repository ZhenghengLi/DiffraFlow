#ifndef FtrConfig_H
#define FtrConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <map>

#include "FeatureExtraction.hh"

using std::string;
using std::map;

namespace diffraflow {
    class FtrConfig : public GenericConfiguration {
    public:
        FtrConfig();
        ~FtrConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        FeatureExtraction::PeakMsseParams peak_msse_params;

        float mean_min_energy;
        float mean_max_energy;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
