import asyncio
import threading
import time
import websockets
import matplotlib.pyplot as pyplot
import matplotlib.image as image
import json
import pyttsx3
import pygame
from tkinter import Tk, Button

pygame.init()
EVENT = pygame.USEREVENT + 1
pygame.mixer.music.set_endevent(EVENT)
engine = pyttsx3.init()


def callback_image():
    global selected_type
    selected_type = 'image'
    root.destroy()


def callback_sound():
    global selected_type
    selected_type = 'sound'
    root.destroy()


selected_type = ''
root = Tk()
root.title("Select the type")
root.geometry("300x100")
Button(text='Image', command=callback_image).pack()
Button(text='Sound', command=callback_sound).pack()
root.mainloop()

fig, ax = pyplot.subplots()
pyplot.ion()


def play():
    while True:
        pygame.mixer.music.play()
        while True:
            if pygame.event.wait().type == EVENT:
                break
        time.sleep(1)  # interval is necessary to load, add a check when file is large


async def handle(websocket, _):
    if selected_type == 'image':
        while True:
            command = int(await websocket.recv())
            settings = json.loads(open('settings.txt').read())
            for setting in settings:
                if setting['command'] == command and setting['type'] == selected_type:
                    ax.imshow(image.imread(setting['path']))
                    pyplot.axis('off')
                    pyplot.pause(0.01)
                    pyplot.axis('off')
                    pyplot.pause(0.01)  # refresh twice because python do not have real multi-thread
                    break
    elif selected_type == 'sound':
        running = False
        while True:
            command = int(await websocket.recv())
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()  # release resource
            settings = json.loads(open('settings.txt').read())
            for setting in settings:
                if setting['command'] == command and setting['type'] == selected_type:
                    engine.save_to_file(setting['text'], 'sound.wav')  # unable to save before released
                    engine.runAndWait()  # use A-B file if text is long
                    break
            pygame.mixer.music.load('sound.wav')
            if not running:
                threading.Thread(target=play).start()
                running = True
    else:
        print('unsupported type')


start_server = websockets.serve(handle, '172.17.6.135', 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
