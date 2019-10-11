package shine;

import java.net.*;
import java.nio.*;
import java.nio.channels.*;

public class Sender {
    InetSocketAddress dest_addr;
    SocketChannel clientSocket;
    ByteBuffer buffer_A;
    ByteBuffer buffer_B;

    int time_threshold;   // by ms
    int size_threshold;   // by byte

    Thread senderThread;

    Sender() {
        senderThread = new Thread(new SenderRunner(this));
        senderThread.start();
    }

    void swap_buffer() {
        ByteBuffer tmp_buffer = buffer_A;
        buffer_A = buffer_B;
        buffer_B = tmp_buffer;
    }

    void set_addr(InetSocketAddress dest_addr) {
        this.dest_addr = dest_addr;
        clientSocket = null;
        time_threshold = 100;
        size_threshold = 512 * 1024;

    }

    void send(long identifier, byte[] data) {

    }

}

class SenderRunner implements Runnable {
    Sender sender;

    SenderRunner(Sender sender) {
        this.sender = sender;
    }
    public void run() {

    }
}