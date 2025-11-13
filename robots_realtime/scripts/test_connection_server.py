import asyncio
import time
import os

# ###############################################################################
# # Server
# ###############################################################################

# class TestServer:
#     def __init__(self, host="0.0.0.0", port=9000):
#         self.host = host
#         self.port = port
#         self.server = None

#     async def start(self):
#         self.server = await asyncio.start_server(self.handle_conn, self.host, self.port)
#         print(f"[SERVER] listening on {self.host}:{self.port}")

#     async def handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
#         addr = writer.get_extra_info("peername")
#         print(f"[SERVER] new connection from {addr}")

#         try:
#             while True:
#                 data = await reader.read(65536)
#                 if not data:
#                     break

#                 # Echo back to client
#                 writer.write(data)
#                 await writer.drain()
#         except Exception as e:
#             print("[SERVER] connection error:", e)

#         writer.close()
#         await writer.wait_closed()
#         print(f"[SERVER] closed {addr}")

#     async def run_forever(self):
#         await self.start()
#         async with self.server:
#             await self.server.serve_forever()


# ###############################################################################
# # Client
# ###############################################################################

# class TestClient:
#     def __init__(self, host="127.0.0.1", port=9000):
#         self.host = host
#         self.port = port
#         self.reader = None
#         self.writer = None

#     async def connect(self):
#         self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
#         print("[CLIENT] connected to server")

#     async def send_and_receive(self, data: bytes) -> bytes:
#         self.writer.write(data)
#         await self.writer.drain()
#         return await self.reader.readexactly(len(data))

#     async def close(self):
#         self.writer.close()
#         await self.writer.wait_closed()
#         print("[CLIENT] connection closed")


# ###############################################################################
# # Throughput test
# ###############################################################################

# async def throughput_test(host = "127.0.0.1", duration_sec=3, packet_size=64 * 1024):
#     """
#     Sends as fast as possible for duration_sec seconds and measures throughput.
#     """

#     client = TestClient(host=host)
#     await client.connect()

#     data = os.urandom(packet_size)
#     sent_bytes = 0

#     start = time.time()
#     while time.time() - start < duration_sec:
#         await client.send_and_receive(data)
#         sent_bytes += len(data)

#     elapsed = time.time() - start
#     mbps = (sent_bytes / elapsed) / 1e6

#     print(f"\n[RESULT] Sent {sent_bytes/1e6:.2f} MB in {elapsed:.2f} sec")
#     print(f"[THROUGHPUT] {mbps:.2f} MB/s\n")

#     await client.close()


# ###############################################################################
# # Entrypoint
# ###############################################################################

# async def main():
#     # Run server in background
#     server = TestServer()
#     asyncio.create_task(server.run_forever())

#     # Wait for server
#     # await asyncio.sleep(0.5)
#     while True:
#         await asyncio.sleep(1)

#     # Run throughput test
#     # await throughput_test(host="100.98.77.125", duration_sec=3, packet_size=128 * 1024)  # try 64 KB, 128 KB, 1 MB, etc.

# if __name__ == "__main__":
#     asyncio.run(main())

class TestServer:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port

    async def start(self):
        server = await asyncio.start_server(self.handle_conn, self.host, self.port)
        print(f"[SERVER] listening on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

    async def handle_conn(self, reader, writer):
        print("[SERVER] client connected")
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                # DO NOT await writer.drain() every time
        except Exception as e:
            print("server error:", e)

async def main():
    # Run server in background
    server = TestServer()
    asyncio.create_task(server.start())

    # Wait for server
    # await asyncio.sleep(0.5)
    while True:
        await asyncio.sleep(1)

    # Run throughput test
    # await throughput_test(host="100.98.77.125", duration_sec=3, packet_size=128 * 1024)  # try 64 KB, 128 KB, 1 MB, etc.

if __name__ == "__main__":
    asyncio.run(main())