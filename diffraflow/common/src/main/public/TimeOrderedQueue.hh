#ifndef TimeOrderedQueue_H
#define TimeOrderedQueue_H

#include <queue>

namespace diffraflow {

    template <typename T, typename Container = std::vector<T>, typename Compare = std::greater<typename Container::value_type> >
    class TimeOrderedQueue: public std::priority_queue<T, Container, Compare> {
    private:
        T tail;
    public:
        void push(const T& value);
        uint64_t distance() const;
    };

    template <typename T, typename Container, typename Compare>
    void TimeOrderedQueue<T, Container, Compare>::push(const T& value) {
        if (std::priority_queue<T, Container, Compare>::empty()) {
            tail = value;
            std::priority_queue<T, Container, Compare>::push(value);
        } else {
            if (std::priority_queue<T, Container, Compare>::comp(value, tail)) {
                tail = value;
            }
            std::priority_queue<T, Container, Compare>::push(value);
        }
    }

    template <typename T, typename Container, typename Compare>
    uint64_t TimeOrderedQueue<T, Container, Compare>::distance() const {
        return tail - std::priority_queue<T, Container, Compare>::top();
    }

}

#endif