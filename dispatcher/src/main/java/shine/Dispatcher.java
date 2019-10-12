package shine;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.*;
import java.net.*;
import java.nio.channels.*;

public class Dispatcher {
    String host;
    int port;
    AtomicInteger conn_counts;
    AtomicInteger max_conn_counts;

    InetSocketAddress[] dest_addresses;
    Sender[] senders;

    Dispatcher(Properties props) {
        host = "localhost";
        port = 1234;
        conn_counts = new AtomicInteger(0);
        max_conn_counts = new AtomicInteger(10);
        // create dest_addresses according to address list

        // create a sender for each address
        senders = new Sender[dest_addresses.length];
        for (int i = 0; i < dest_addresses.length; i++) {
            senders[i].set_addr(dest_addresses[i]);
            try {
                senders[i].connect();
            } catch (IOException e) {
                System.out.println("Failed to connect to " + dest_addresses[i] + " : " + e);
            }
            senders[i].start_runner();
        }

    }

    public void start() throws IOException{
        ServerSocketChannel ssc = ServerSocketChannel.open();
        ssc.configureBlocking(true);
        ssc.socket().bind(new InetSocketAddress(host, port));
        while (true) {
            // waiting connection
            SocketChannel clientSocket = ssc.accept();
            clientSocket.configureBlocking(true);
            // create a dedicated thread to handle this connection
            new Thread(new ImageConnection(clientSocket, this)).start();
        }
    }

    public static void main(String[] args) {
        System.out.println("Starting Dispatcher ...");

    }

}