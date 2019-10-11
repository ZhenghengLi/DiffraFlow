package shine;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.net.*;
import java.nio.*;
import java.nio.channels.*;

public class ReceivingServer {
    Selector clientSelector;
    String host;
    int port;

    Properties kafka_props;

    Executor executor_in;
    Executor executor_out;

    ReceivingServer(Properties props) {
        this.host = props.getProperty("listen.host", "localhost");
        this.port =  Integer.parseInt(props.getProperty("listen.port", "1234"));
        int threads = Integer.parseInt(props.getProperty("worker.threads", "4"));
        executor_in = Executors.newFixedThreadPool(threads);
        executor_out = Executors.newFixedThreadPool(threads);
        kafka_props = new Properties();
        kafka_props.setProperty("bootstrap.servers", props.getProperty("kafka.bootstrap.servers", "localhost:9092"));
    }

    public void start() throws IOException{
        ServerSocketChannel ssc;
        clientSelector = Selector.open();
        ssc = ServerSocketChannel.open();
        ssc.configureBlocking(false);
        ssc.socket().bind(new InetSocketAddress(host, port));
        ssc.register(clientSelector, SelectionKey.OP_ACCEPT);
        while (true) {
            while (clientSelector.select(100) == 0);
            Set<SelectionKey> readySet = clientSelector.selectedKeys();
            for (Iterator<SelectionKey> it = readySet.iterator(); it.hasNext(); it.remove()) {
                final SelectionKey key = it.next();
                if (key.isAcceptable()) {
                    SocketChannel clientSocket = ssc.accept();
                    System.out.println("One client come in: " + clientSocket.toString());
                    executor_in.execute(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                acceptClient(clientSocket);
                            } catch (IOException e) {
                                System.out.println(e);
                            }                           
                        }
                    });
                } else {
                    key.interestOps(0);
                    executor_in.execute(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                handleClient(key);
                            } catch (IOException e) {
                                System.out.println(e);
                            }
                        }
                    });
                }
            }
        }
    }

    void acceptClient(SocketChannel clientSocket) throws IOException {
        // Greeting
        clientSocket.configureBlocking(true);
        ByteBuffer buffer = ByteBuffer.allocateDirect(12);
        while (buffer.hasRemaining()) { clientSocket.read(buffer); }
        buffer.flip();
        int head = buffer.getInt();
        int size = buffer.getInt();
        // use 0xAABBCCDD as the greeting message header
        if (head != 0xAABBCCDD || size != 4) {
            buffer.clear();
            buffer.putInt(321);
            buffer.flip();
            clientSocket.write(buffer);
            clientSocket.close();
            System.out.println("Got wrong greeting message from client, close connection.");
            return;
        }
        // connect to kafka
        Dispatcher dsp = new Dispatcher();
        // configure dispatcher here
        if (!dsp.connect(kafka_props)) {
            buffer.clear();
            buffer.putInt(456);
            buffer.flip();
            clientSocket.write(buffer);
            clientSocket.close();
            System.out.println("Failed to connect to Kafka, close connection.");
            return;
        } else {
            System.out.println("Two sides connection is successful for client.");
        }
        int detid = buffer.getInt();
        System.out.println("Detector ID: " + detid);
        buffer.clear();
        buffer.putInt(123); // here use 123 as the successful connection code.
        buffer.flip();
        clientSocket.write(buffer);
        // prepare for non-blocking data receiving
        clientSocket.configureBlocking(false);
        SelectionKey key = clientSocket.register(clientSelector, SelectionKey.OP_READ);
        BlockingQueue<ImagePacket> msgQueue = new LinkedBlockingQueue<ImagePacket>(1024);
        key.attach(new ImageConnection(clientSocket, msgQueue));
        executor_out.execute(new Runnable() {
            public void run() {
                while (clientSocket.isConnected() || !msgQueue.isEmpty()) {
                    dsp.dispatch_one(msgQueue);
                }
            }
        });
    }

    void handleClient(SelectionKey key) throws IOException {
        // System.out.println("handleClient");
        ImageConnection client = (ImageConnection) key.attachment();
        if (key.isReadable()) {
            client.read(key);
        } else {
            client.write(key);
        }
        clientSelector.wakeup();
    }

}
