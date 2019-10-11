package shine;

import java.util.*;

public class ImagePacket {

    public byte[] data;
    public long identifier;

    ImagePacket(long identifier, byte[] data) {
        this.identifier = identifier;
        this.data = data;
    }

    public void print() {
        System.out.println("identifier: " + identifier);
        System.out.println("image data: [" + new String(data) + "]");
    }

}