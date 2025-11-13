import asyncio
import numpy as np
import msgpack
import msgpack_numpy as m
import struct

m.patch()  # allow numpy arrays

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


class MsgpackNumpyServer:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port

    async def start(self):
        server = await asyncio.start_server(self.handle, self.host, self.port)
        print(f"[SERVER] listening on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

    async def handle(self, reader, writer):
        print("[SERVER] client connected")
        try:
            while True:
                # read msgpack numpy dict
                request = await recv_framed(reader)

                # process request → this is where or transform happens
                response = self.process(request)

                # send response back
                await send_framed(writer, response)

        except asyncio.IncompleteReadError:
            print("[SERVER] client disconnected")
        except Exception as e:
            print("[SERVER] error:", e)

    def process(self, req: dict) -> dict:
        # Example: echo back shapes and metadata
        out = {}
        for k, v in req.items():
            if hasattr(v, "shape"):
                out[k] = {"shape": v.shape, "dtype": str(v.dtype)}
            else:
                out[k] = v
        return out

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

    async def send_request(self, data: dict) -> dict:
        await send_framed(self.writer, data)
        response = await recv_framed(self.reader)
        return response

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
