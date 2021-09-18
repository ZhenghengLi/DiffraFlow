#ifndef FtrConfig_H
#define FtrConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <map>

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
        float peak_min_energy;
        float peak_max_energy;
        float peak_inlier_thr;
        float peak_outlier_thr;
        float peak_residual_thr;
        float peak_energy_thr;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
