package shine;

import java.io.*;
import java.util.concurrent.*;
import java.nio.*;
import java.nio.channels.*;

public class ImageConnection {
    SocketChannel clientSocket;
    BlockingQueue<ImagePacket> msgQueue;
    ByteBuffer buffer_A = ByteBuffer.allocateDirect(1024 * 1024);
    ByteBuffer buffer_B = ByteBuffer.allocateDirect(1024 * 1024);

    void swap_buffer() {
        ByteBuffer tmp_buffer = buffer_A;
        buffer_A = buffer_B;
        buffer_B = tmp_buffer;
    }

    ImageConnection(SocketChannel clientSocket, BlockingQueue<ImagePacket> msgQueue) {
        this.clientSocket = clientSocket;
        this.msgQueue = msgQueue;
    }

    void read(SelectionKey key) throws IOException {
        long read_size = clientSocket.read(buffer_A);
        if (read_size < 0) {
            System.out.println("Connection is closed by the client");
            clientSocket.close();
            key.cancel();
            return;
        }
        // System.out.println("read_size: " + read_size);
        if (read_size > 0) {
            process(key);
        } else {
            key.interestOps(SelectionKey.OP_READ);
        }
    }

    void process(SelectionKey key) throws IOException {
        buffer_A.flip();
        while (true) {
            if (buffer_A.remaining() <= 8) {
                buffer_B.put(buffer_A);
                buffer_A.clear();
                swap_buffer();
                key.interestOps(SelectionKey.OP_READ);
                break;
            }
            int head = buffer_A.getInt();
            int size = buffer_A.getInt();
            if (buffer_A.remaining() < size) {
                buffer_A.position(buffer_A.position() - 8);
                buffer_B.put(buffer_A);
                buffer_A.clear();
                swap_buffer();
                key.interestOps(SelectionKey.OP_READ);
                break;
            }
            if (head == 0xABCDEEFF) { // header of image data
                if (size < 8) {
                    System.out.println("got wrong image packet, close the connection.");
                    clientSocket.close();
                    key.cancel();
                    return;
                }
                long identifier = buffer_A.getLong();  // => key
                byte[] data_arr = new byte[size - 8];  // => value
                buffer_A.get(data_arr);
                ImagePacket img = new ImagePacket(identifier, data_arr);
                if (msgQueue.offer(img)) {
                    // System.out.println("one image is pushed into queue.");
                } else {
                    System.out.println("Failed to add image: " + new String(data_arr));
                }
            }
        }
    }

    void write(SelectionKey key) throws IOException {

    }

}