import asyncio
import time
import os

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

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())