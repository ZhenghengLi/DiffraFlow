#include "GeneralConnection.hh"
#include "Decoder.hh"

#include <cassert>
#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>

using std::cout;
using std::cerr;
using std::endl;
using std::copy;

shine::GeneralConnection::GeneralConnection(int sock_fd, size_t buff_sz, size_t pkt_ml, uint32_t greet_hd) {
    assert(sock_fd > 0);
    client_sock_fd_ = sock_fd;
    buffer_size_ = buff_sz;
    buffer_ = new char[buffer_size_];
    pkt_maxlen_ = pkt_ml;
    greeting_head_ = greet_hd;
    slice_begin_ = 0;
    done_flag_ = false;
    connection_id_ = -1;
}

shine::GeneralConnection::~GeneralConnection() {
    delete [] buffer_;
}

void shine::GeneralConnection::run() {
    if (start_connection_()) {
        before_transferring_();
        while (!done_flag_ && do_transferring_());
    }
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool shine::GeneralConnection::done() {
    return done_flag_;
}

void shine::GeneralConnection::shift_left_(const size_t position, const size_t limit) {
    if (position == 0) {
        copy(buffer_ + position, buffer_ + limit, buffer_);
    }
    slice_begin_ = limit - position;
}

bool shine::GeneralConnection::start_connection_() {
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            cout << "socket is closed by the client." << endl;
            return false;
        }
        if (slice_begin_ + slice_length < 12) {
            slice_begin_ += slice_length;
        } else {
            break;
        }
    }
    uint32_t success_code = htonl(200);
    uint32_t failure_code = htonl(300);
    uint32_t head = gDC.decode_byte<int32_t>(buffer_, 0, 3);
    uint32_t size = gDC.decode_byte<int32_t>(buffer_, 4, 7);
    if (head != greeting_head_ || size != 4) {
        cout << "got wrong greeting message, close the connection." << endl;
        write(client_sock_fd_, &failure_code, 4);
        done_flag_ = false;
        return false;
    }
    connection_id_ = gDC.decode_byte<int32_t>(buffer_, 8, 11);
    write(client_sock_fd_, &success_code, 4);
    // ready for transferring data
    slice_begin_ = 0;
    return true;
}