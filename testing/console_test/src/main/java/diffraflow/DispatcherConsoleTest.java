package diffraflow;

import java.net.*;
import java.io.*;
import java.util.*;

public class DispatcherConsoleTest {
    public static void main(String[] args) throws IOException{
        if (args.length < 1) {
            System.out.println("please input the host and port.");
            return;
        }
        System.out.println("Start test on " + args[0] + ":" + args[1] + " ...");

        Socket sock = new Socket(args[0], Integer.parseInt(args[1]));
        DataInputStream in = new DataInputStream(sock.getInputStream());
        DataOutputStream out = new DataOutputStream(sock.getOutputStream());
        int packet_head = 0xFFDD1234;
        int packet_size = 4;
        out.writeInt(packet_head);
        out.writeInt(packet_size);
        out.writeInt(1614);
        out.flush();
        int res = in.readInt();
        System.out.println("response code: " + res);

        InputStreamReader chars_in = new InputStreamReader(System.in);
        BufferedReader buff_in =  new BufferedReader(chars_in);

        Random rand = new Random();

        packet_head = 0xFFF22DDD;
        int payload_type = 0xABCDFFFF;
        while (true) {
            String line = buff_in.readLine();
            if (line.length() == 0) continue;
            if (line == "quit") break;
            System.out.println(line + " : " + line.getBytes().length);
            byte[] data = line.getBytes();
            out.writeInt(packet_head);
            out.writeInt(data.length + 4 + 8);
            out.writeInt(payload_type);
            long identifier = rand.nextLong();
            System.out.println("identifier: " + identifier);
            out.writeLong(identifier);
            out.write(data);
            out.flush();
        }

        sock.close();

    }

}
