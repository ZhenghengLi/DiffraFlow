package shine;

import java.io.IOException;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.concurrent.atomic.*;

public class Sender {
    InetSocketAddress dest_addr;
    SocketChannel clientSocket;
    ByteBuffer buffer_A;
    ByteBuffer buffer_B;

    long time_threshold;   // by ms
    long size_threshold;   // by byte

    AtomicBoolean run;
    Thread senderRunner;

    Sender() {
        time_threshold = 100;
        size_threshold = 512 * 1024;
        buffer_A = ByteBuffer.allocateDirect(1024 * 1024);
        buffer_B = ByteBuffer.allocateDirect(1024 * 1024);
        senderRunner = new Thread(new SenderRunner(this));
    }

    public void set_addr(InetSocketAddress dest_addr) {
        this.dest_addr = dest_addr;
        clientSocket = null;

    }

    public void start_runner() {
        run.set(true);
        senderRunner.start();
    }

    synchronized public void stop_runner() {
        run.set(false);
        try {
            notify();
            senderRunner.join();
        } catch (InterruptedException e) {
            System.out.println(e);
        }

    }

    public void connect() throws IOException {
        clientSocket = SocketChannel.open(dest_addr);
        clientSocket.configureBlocking(true);
        // send greeting message
        ByteBuffer buffer = ByteBuffer.allocateDirect(12);
        buffer.putInt(0xAAAABBBB);
        buffer.putInt(0);
        clientSocket.write(buffer);
        buffer.clear();
        clientSocket.read(buffer);
        int response_code = buffer.getInt();
        if (response_code != 200) {
            clientSocket.close();
            clientSocket = null;
            throw new IOException("Got wrong response code.");
        }
    }

    public void close() {
        try {
            if (clientSocket != null) {
                clientSocket.close();
            }
        } catch (IOException e) {
            System.out.println(e);
        } finally {
            clientSocket = null;
        }
    }

    synchronized public void send(long identifier, byte[] data) {
        if (buffer_A.remaining() < 16 + data.length) {
            System.out.println("WARNING: buffer is full.");
            return;
        }
        buffer_A.putInt(0xABCDEEFF);
        buffer_A.putInt(8 + data.length);
        buffer_A.putLong(identifier);
        buffer_A.put(data);
        if (buffer_A.position() > size_threshold) {
            notify();
        }
    }

    synchronized public boolean swap() {
        try {
            wait(time_threshold);
        } catch(Exception e) {
            System.out.println(e);
            return false;
        }
        if (buffer_A.position() > 0) {
            ByteBuffer tmp_buffer = buffer_B;
            buffer_B = buffer_A;
            buffer_A = tmp_buffer;
            return true;
        } else {
            return false;
        }
    }
}

class SenderRunner implements Runnable {
    Sender sender;

    SenderRunner(Sender sender) {
        this.sender = sender;
    }

    public void run() {
        while (sender.run.get()) {
            if (sender.swap()) {
                send_buffer(sender.buffer_B);
            }
        }
    }

    void send_buffer(ByteBuffer buffer) {
        if (sender.clientSocket == null || !sender.clientSocket.isConnected()) {
            try {
                sender.connect();
            } catch (IOException e) {
                System.out.println(e);
                sender.close();
                return;
            }
        } else {
            try {
                buffer.flip();
                sender.clientSocket.write(buffer);
                buffer.clear();
            } catch (IOException e) {
                System.out.println(e);
                sender.close();
                return;
            }
        }
    }
}