"""msgpack + numpy serialization used by Publisher and Subscriber."""

from __future__ import annotations

import numpy as np
import msgpack
import msgpack_numpy


def pack(data: dict) -> bytes:
    return msgpack.packb(data, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)
