import asyncio
import time
import os

class AsyncThroughputClient:
    def __init__(self, host, port=9000):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print("[CLIENT] connected")

    async def sender(self, payload, duration):
        end = time.time() + duration
        sent = 0

        # Allow large write buffers
        transport = self.writer.transport
        transport.set_write_buffer_limits(0)

        while time.time() < end:
            self.writer.write(payload)
            sent += len(payload)

            # occasionally yield control & flush
            if sent % (8*1024*1024) < len(payload):
                await self.writer.drain()

        return sent

    async def receiver(self, nbytes):
        received = 0
        while received < nbytes:
            chunk = await self.reader.read(65536)
            if not chunk:
                break
            received += len(chunk)
        return received

    async def run_test(self, duration=3, packet_size=1024*256):
        payload = os.urandom(packet_size)

        await self.connect()

        # sender task: blast data
        sender_task = asyncio.create_task(self.sender(payload, duration))

        # receiver task: drain echo stream
        recv_task = asyncio.create_task(
            self.receiver(float('inf'))  # receive forever
        )

        sent = await sender_task
        await asyncio.sleep(0.1)

        # close so receiver ends
        self.writer.close()
        await self.writer.wait_closed()

        recvd = recv_task.result()

        print(f"[RESULT] sent={sent/1e6:.2f}MB  recv={recvd/1e6:.2f}MB")
        print(f"[THROUGHPUT] {(sent/duration)/1e6:.2f} MB/s (raw send rate)")


###############################################################################
# Entrypoint
###############################################################################

async def main():
    # Run server in background
    # server = TestServer()
    # asyncio.create_task(server.run_forever())

    # Wait for server
    # await asyncio.sleep(0.5)

    # Run throughput test
    # await throughput_test(host="128.32.175.42", duration_sec=3, packet_size=128 * 1024 * 5)  # try 64 KB, 128 KB, 1 MB, etc.

    # client = AsyncThroughputClient(host="128.32.175.42")
    client = AsyncThroughputClient(host="0.0.0.0")
    await client.run_test(duration=3, packet_size=128 * 1024 * 10)

if __name__ == "__main__":
    asyncio.run(main())
