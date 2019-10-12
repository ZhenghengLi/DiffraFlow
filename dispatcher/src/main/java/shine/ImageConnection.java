package shine;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;

public class ImageConnection implements Runnable {
    SocketChannel clientSocket;
    Dispatcher dispatcher;
    ByteBuffer buffer_A = ByteBuffer.allocateDirect(1024 * 1024);
    ByteBuffer buffer_B = ByteBuffer.allocateDirect(1024 * 1024);


    void swap_buffer() {
        ByteBuffer tmp_buffer = buffer_A;
        buffer_A = buffer_B;
        buffer_B = tmp_buffer;
    }

    ImageConnection(SocketChannel clientSocket, Dispatcher dispatcher) {
        this.clientSocket = clientSocket;
        this.dispatcher = dispatcher;
    }

    public void run() {
        try {
            if (!start_greeting()) return;
        } catch (IOException e) {
            System.out.println(e);
            return;
        }
        dispatcher.conn_counts.incrementAndGet();
        try {
            while (transfering());
        } catch (IOException e) {
            System.out.println(e);
        }
        dispatcher.conn_counts.decrementAndGet();
    }

    boolean start_greeting() throws IOException {
        ByteBuffer buffer = ByteBuffer.allocateDirect(12);
        while (buffer.hasRemaining()) { clientSocket.read(buffer); }
        buffer.flip();
        int head = buffer.getInt();
        int size = buffer.getInt();
        // use 0xAABBCCDD as the greeting message header
        if (head != 0xAABBCCDD || size != 4) {
            System.out.println("Got wrong greeting message from client, close connection.");
            buffer.clear();
            buffer.putInt(321);
            buffer.flip();
            clientSocket.write(buffer);
            clientSocket.close();
            return false;
        }
        if (dispatcher.conn_counts.get() > dispatcher.max_conn_counts.get()) {
            System.out.println("Reach the maximum connection counts, close connection.");
            buffer.clear();
            buffer.putInt(999);
            buffer.flip();
            clientSocket.write(buffer);
            clientSocket.close();
            return false;
        }
        int detid = buffer.getInt();
        System.out.println("Detector ID: " + detid);
        buffer.clear();
        buffer.putInt(123); // here use 123 as the successful connection code.
        buffer.flip();
        clientSocket.write(buffer);
        return true;
    }

    boolean transfering() throws IOException {
        long read_size = clientSocket.read(buffer_A);
        if (read_size < 0) {
            System.out.println("Connection is closed by the client");
            clientSocket.close();
            return false;
        }
        buffer_A.flip();
        while (true) {
            if (buffer_A.remaining() <= 8) {
                buffer_B.put(buffer_A);
                buffer_A.clear();
                swap_buffer();
                break;
            }
            int head = buffer_A.getInt();
            int size = buffer_A.getInt();
            if (buffer_A.remaining() < size) {
                buffer_A.position(buffer_A.position() - 8);
                buffer_B.put(buffer_A);
                buffer_A.clear();
                swap_buffer();
                break;
            }
            if (head == 0xABCDEEFF) { // header of image data
                if (size < 8) {
                    System.out.println("got wrong image packet, close the connection.");
                    clientSocket.close();
                    return false;
                }
                long identifier = buffer_A.getLong();  // => key
                byte[] data_arr = new byte[size - 8];  // => value
                buffer_A.get(data_arr);
                int index = Long.hashCode(identifier) % dispatcher.senders.length;
                dispatcher.senders[index].send(identifier, data_arr);
            }
        }
        return true;
    }
}