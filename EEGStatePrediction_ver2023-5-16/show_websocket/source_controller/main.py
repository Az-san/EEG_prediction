import asyncio
import websockets


async def command():
    async with websockets.connect('ws://127.0.0.1:5678') as websocket:
        while True:
            await websocket.send("0")
            print("0")
            await asyncio.sleep(3)
            await websocket.send("1")
            print("1")
            await asyncio.sleep(3)


asyncio.run(command())
