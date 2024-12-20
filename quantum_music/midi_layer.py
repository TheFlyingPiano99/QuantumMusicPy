import mido     # Source: https://mido.readthedocs.io/en/stable/index.html
import pathlib

class MidiData:
    def __init__(self, file_path: pathlib.Path):
        file = mido.MidiFile(file_path)
        for i, track in enumerate(file.tracks):
            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                print(msg)

