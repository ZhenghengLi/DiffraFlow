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

    Dispatcher(Properties props) {
        host = "localhost";
        port = 1234;
        conn_counts = new AtomicInteger(0);
        max_conn_counts = new AtomicInteger(10);

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
}