#ifndef TimeOrderedQueue_H
#define TimeOrderedQueue_H

#include <queue>

namespace diffraflow {

    template <typename ET, typename DT, typename Container = std::vector<ET>, typename Compare = std::greater<typename Container::value_type> >
    class TimeOrderedQueue: public std::priority_queue<ET, Container, Compare> {
    private:
        ET tail;
    public:
        void push(const ET& value);
        DT distance() const;
    };

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
        return tail - std::priority_queue<ET, Container, Compare>::top();
    }

}

#endif