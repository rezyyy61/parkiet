import asyncio
import json
import time
import websockets

URL = "ws://127.0.0.1:8888/ws/tts"

async def main():
    text = "Goedemiddag Jan, met Bart van Circle en Borne. Mag ik je even kort storen?"

    started = time.monotonic()

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({
            "text": text,
            "voice": "s2_plain_t10",
        }))

        first_audio = None
        chunks = 0
        total_bytes = 0

        while True:
            msg = await ws.recv()

            if isinstance(msg, bytes):
                chunks += 1
                total_bytes += len(msg)

                if first_audio is None:
                    first_audio = time.monotonic()
                    print(f"TTFA_MS={(first_audio - started) * 1000:.0f}")

            else:
                print(msg)

                try:
                    data = json.loads(msg)
                    if data.get("type") == "done":
                        break
                except Exception:
                    pass

        print(f"CHUNKS={chunks}")
        print(f"BYTES={total_bytes}")
        print(f"TOTAL_MS={(time.monotonic() - started) * 1000:.0f}")

asyncio.run(main())
