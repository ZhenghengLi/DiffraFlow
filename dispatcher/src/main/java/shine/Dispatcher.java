package shine;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.*;
import java.net.*;
import java.nio.channels.*;
import java.nio.file.*;
import org.apache.commons.cli.*;

public class Dispatcher {
    String host;
    int port;
    int sender_id;
    AtomicInteger conn_counts;
    AtomicInteger max_conn_counts;

    Sender[] senders;

    Dispatcher(Properties props) {
        this.host = props.getProperty("listen.host", "0.0.0.0");
        this.port = Integer.parseInt(props.getProperty("listen.port", "1234"));
        this.sender_id = Integer.parseInt(props.getProperty("sender.id", "1614"));
        this.conn_counts = new AtomicInteger(0);
        this.max_conn_counts = new AtomicInteger(10);
        senders = null;
    }

    private List<InetSocketAddress> read_address_list_(String address_list_fn) {
        List<InetSocketAddress> addresses = new ArrayList<>();
        try {
            List<String> allLines = Files.readAllLines(Paths.get(address_list_fn));
            for (String line: allLines) {
                String[] host_port_pair = line.split(":");
                if (host_port_pair.length != 2) continue;
                addresses.add(new InetSocketAddress(host_port_pair[0].trim(), Integer.parseInt(host_port_pair[1])));
            }
            if (addresses.size() > 0) {
                return addresses;
            } else {
                System.out.println("read nothing from address list file: " + address_list_fn);
                return null;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public boolean init(String address_list_fn) {
        List<InetSocketAddress> dest_addresses = read_address_list_(address_list_fn);
        if (dest_addresses == null) {
            System.out.println("Failed to read address list file");
            return false;
        }
        // create a sender for each address
        // and try the first connection
        senders = new Sender[dest_addresses.size()];
        for (int i = 0; i < dest_addresses.size(); i++) {
            senders[i].set_addr(dest_addresses.get(i));
            senders[i].set_id(sender_id);
            try {
                senders[i].connect();
            } catch (IOException e) {
                System.out.println("Failed to connect to " + dest_addresses.get(i) + " : " + e);
            }
            senders[i].start_runner();
        }
        return true;
    }

    public void start() throws IOException{
        if (senders == null) {
            System.out.println("The dispatcher server is not initialized.");
            return;
        }
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
        Options options = new Options();
        options.addOption("a", "addrList", true, "address list file");
        options.addOption("c", "config", true, "configuration file");
        options.addOption("h", "help", false, "show help information");
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formater = new HelpFormatter();
        String cmdLineSyntax = "Dispacher [options]";
        CommandLine cmdLine;
        try {
            cmdLine = parser.parse(options, args);
            if (cmdLine.hasOption("help")) {
                formater.printHelp(cmdLineSyntax, options);
                return;
            }
        } catch (ParseException e) {
            formater.printHelp(cmdLineSyntax, options);
            System.exit(2);
            return;
        }
        Properties props = new Properties();
        // load user-defined properties
        String configFileName = cmdLine.getOptionValue("config");
        try {
            FileInputStream fin = new FileInputStream(configFileName);
            props.load(fin);
            fin.close();
        } catch (IOException e) {
            System.out.println(e);
            System.exit(3);
            return;
        }
        // start dispatcher server
        Dispatcher dispatcher = new Dispatcher(props);
        if (!dispatcher.init(cmdLine.getOptionValue("addrList"))) {
            System.out.println("Failed to initialize dispatcher server.");
            System.exit(3);
            return;
        }
        try {
            dispatcher.start();
        } catch(IOException e) {
            System.out.println(e);
        }
    }
}