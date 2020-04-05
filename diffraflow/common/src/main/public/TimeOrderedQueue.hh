#ifndef TimeOrderedQueue_H
#define TimeOrderedQueue_H

#include <queue>

namespace diffraflow {

    template <typename ET, typename DT, typename Container = std::vector<ET>, typename Compare = std::greater<typename Container::value_type> >
    class TimeOrderedQueue: public std::priority_queue<ET, Container, Compare> {
    public:
        TimeOrderedQueue();
        void push(const ET& value);
        DT distance() const;
    private:
        ET tail;
    };

    template <typename ET, typename DT, typename Container, typename Compare>
    TimeOrderedQueue<ET, DT, Container, Compare>::TimeOrderedQueue(): std::priority_queue<ET, Container, Compare>() {

    }

    template <typename ET, typename DT, typename Container, typename Compare>
    void TimeOrderedQueue<ET, DT, Container, Compare>::push(const ET& value) {
        if (std::priority_queue<ET, Container, Compare>::empty()) {
            tail = value;
            std::priority_queue<ET, Container, Compare>::push(value);
        } else {
            if (std::priority_queue<ET, Container, Compare>::comp(value, tail)) {
                tail = value;
            }
            std::priority_queue<ET, Container, Compare>::push(value);
        }
    }

    template <typename ET, typename DT, typename Container, typename Compare>
    DT TimeOrderedQueue<ET, DT, Container, Compare>::distance() const {
        if (std::priority_queue<ET, Container, Compare>::empty()) {
            return DT();
        } else {
            return tail - std::priority_queue<ET, Container, Compare>::top();
        }
    }

}

#endif