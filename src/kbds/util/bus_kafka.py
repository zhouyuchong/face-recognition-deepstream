from kafka import KafkaProducer
import json

def init_bus_kafka(config):
    conn_str = config.split(";")[:2]
    symbol = ":"
    conn_str = symbol.join(conn_str)
    # print(conn_str)
    producer = KafkaProducer(bootstrap_servers=[conn_str])
    return producer
