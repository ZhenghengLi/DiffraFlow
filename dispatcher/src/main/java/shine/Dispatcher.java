package shine;

import java.io.*;
import java.util.concurrent.*;
import java.util.*;

import org.apache.commons.cli.*;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.serialization.ByteArraySerializer;
import org.apache.kafka.common.serialization.LongSerializer;

public class Dispatcher {

    KafkaProducer<Long, byte[]> kafka_producer;
    String topic;

    public Boolean connect(Properties props) {

        // for debug
        return true;

        // send an empty record to kafka to check connection
        // ---- code block begin ----
        // topic = props.getProperty("kafka.topic", "test123");
        // Long key = new Random().nextLong();
        // byte[] value = new byte[0];
        // kafka_producer = new KafkaProducer<>(props, new LongSerializer(), new ByteArraySerializer());
        // ProducerRecord<Long, byte[]> empty_record = new ProducerRecord<>(topic, key, value);
        // try {
        //     kafka_producer.send(empty_record).get();
        // } catch (Exception e) {
        //     return false;
        // }
        // return true;
        // ---- code block end ----

    }

    public void dispatch_one(BlockingQueue<ImagePacket> msgQueue) {
        ImagePacket img;
        try {
            img = msgQueue.take();
        } catch (InterruptedException e) {
            System.out.println(e);
            return;
        }
        // System.out.println("One image is dispatched: ");

        // for debug
        img.print();
        return;

        // send data to kafka
        // ---- code block begin ----
        // ProducerRecord<Long, byte[]> record = new ProducerRecord<>(topic, img.identifier, img.data);
        // kafka_producer.send(record, new ErrorHandler());
        // ---- code block end ----

    }

    public static void main(String[] args) {

        Options options = new Options();
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
        if (cmdLine.hasOption("config")) {
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
        }

        // start image data receiving server
        ReceivingServer recSrv = new ReceivingServer(props);
        System.out.println("Starting Receiving Server ...");
        try {
            recSrv.start();
        } catch (IOException e) {
            System.out.println(e);
        }
    }

}

class ErrorHandler implements Callback {
    @Override
    public void onCompletion(RecordMetadata recordMetadata, Exception e) {
        if (e != null) {
            e.printStackTrace();
        }
    }
}