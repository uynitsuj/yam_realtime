import asyncio
import time
import numpy as np
import msgpack
import msgpack_numpy as m
import struct

m.patch()  # enable numpy support


def encode_msg(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)

def decode_msg(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=True)

async def send_framed(writer, obj: dict):
    payload = encode_msg(obj)
    header = struct.pack("!I", len(payload))
    writer.write(header + payload)
    await writer.drain()

async def recv_framed(reader):
    header = await reader.readexactly(4)
    (msg_len,) = struct.unpack("!I", header)
    payload = await reader.readexactly(msg_len)
    return decode_msg(payload)


class MsgpackNumpyClient:
    def __init__(self, host="127.0.0.1", port=9000):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print("[CLIENT] connected")
        return self

    async def send_request(self, data: dict):
        t0 = time.perf_counter()
        await send_framed(self.writer, data)
        response = await recv_framed(self.reader)
        t1 = time.perf_counter()
        return response, (t1 - t0)

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()


async def run_test(host="127.0.0.1", port=9000, 
                   duration=3, shape=(720, 1280, 3)):
    client = MsgpackNumpyClient(host, port)
    await client.connect()

    sent_bytes = 0
    start = time.time()

    counter = 0

    while time.time() - start < duration:
        arr = np.random.randint(0, 255, shape, dtype=np.uint8)

        msg = {"image": arr}
        raw = encode_msg(msg)
        sent_bytes += len(raw)

        resp = await client.send_request(msg)
        counter += 1

    elapsed = time.time() - start
    mbps = sent_bytes / elapsed / 1e6

    print(f"[RESULT] Sent {sent_bytes/1e6:.2f} MB in {elapsed:.2f} sec")
    print(f"[THROUGHPUT] {mbps:.2f} MB/s")
    print(f"[COUNTER] Sent {counter} messages")
    print(f"Last server response: {resp}")

    await client.close()

async def latency_test(host="127.0.0.1", port=9000, trials=200):
    client = MsgpackNumpyClient(host, port)
    await client.connect()

    latencies = []

    for _ in range(trials):
        msg = {"ping": 1}
        _, dt = await client.send_request(msg)
        latencies.append(dt)

    await client.close()

    latencies_ms = [l * 1000 for l in latencies]

    print("\n===== LATENCY RESULTS =====")
    print(f"min: {min(latencies_ms):.3f} ms")
    print(f"avg: {sum(latencies_ms)/len(latencies_ms):.3f} ms")
    print(f"max: {max(latencies_ms):.3f} ms")
    print(f"std: {np.std(latencies_ms):.3f} ms")
    print("===========================\n")

if __name__ == "__main__":
    # Change host to remote IP (e.g., 128.32.175.97) for real test
    asyncio.run(run_test(host="128.32.175.97"))
    asyncio.run(latency_test(host="128.32.175.97"))
