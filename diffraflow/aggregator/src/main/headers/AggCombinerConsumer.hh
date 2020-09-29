#ifndef __AggCombinerConsumer_H__
#define __AggCombinerConsumer_H__

#include "AggBaseConsumer.hh"
#include <cpprest/json.h>

using namespace web;

namespace diffraflow {

    class AggMetrics;

    class AggCombinerConsumer : public AggBaseConsumer {
    public:
        explicit AggCombinerConsumer(AggMetrics* metrics);
        ~AggCombinerConsumer();

    protected:
        void process_message_(const pulsar::Message& message) override;

    private:
        json::value simplify_metrics_(const json::value& metrics_json);

    private:
        AggMetrics* aggregated_metrics_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif