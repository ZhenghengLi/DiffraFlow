package diffraflow;

import java.net.*;
import java.io.*;
import java.util.*;

public class DispatcherConsoleTest {
    public static void main(String[] args) throws IOException{
        System.out.println("Start test ...");
        
        Socket sock = new Socket("localhost", 1234);
        DataInputStream in = new DataInputStream(sock.getInputStream());
        DataOutputStream out = new DataOutputStream(sock.getOutputStream());
        int head = 0xAABBCCDD;
        int size = 4;
        out.writeInt(head);
        out.writeInt(size);
        out.writeInt(1614);
        out.flush();
        int res = in.readInt();
        System.out.println("response code: " + res);

        InputStreamReader chars_in = new InputStreamReader(System.in);
        BufferedReader buff_in =  new BufferedReader(chars_in);

        Random rand = new Random();

        head = 0xABCDEEFF;
        while (true) {
            String line = buff_in.readLine();
            if (line.length() == 0) continue;
            if (line == "quit") break;
            System.out.println(line + " : " + line.getBytes().length);
            byte[] data = line.getBytes();
            out.writeInt(head);
            out.writeInt(data.length + 8);
            long identifier = rand.nextLong();
            System.out.println("identifier: " + identifier);
            out.writeLong(identifier);
            out.write(data);
            out.flush();
        }

        sock.close();

    }

}
