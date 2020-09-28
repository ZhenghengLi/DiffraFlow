#ifndef __AggSenderConsumer_H__
#define __AggSenderConsumer_H__

#include "AggBaseConsumer.hh"

namespace diffraflow {

    class AggMetrics;

    class AggSenderConsumer : public AggBaseConsumer {
    public:
        AggSenderConsumer(string name, AggMetrics* metrics);
        ~AggSenderConsumer();

    protected:
        void process_message_(const pulsar::Message& message) override;

    private:
        AggMetrics* aggregated_metrics_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif