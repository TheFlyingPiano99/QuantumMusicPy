import mido     # Source: https://mido.readthedocs.io/en/stable/index.html
import pathlib
import time
import quantum_music.music_layer as music


class MidiTrack:
    __ticks_per_beat: int
    __latest_tempo_bpm: int
    __latest_velocity: int
    __track: mido.MidiTrack
    __time_signature: music.TimeSignature
    __current_rest_length: int

    def __init__(self, file_path: pathlib.Path = pathlib.Path(""), track_idx=0):
        self.__ticks_per_beat = 480
        self.__latest_tempo_bpm = 120
        self.__latest_velocity = 64
        self.__current_rest_length = 0
        self.__time_signature = music.TimeSignature()
        self.__time_signature_msg: mido.MetaMessage

        if file_path == "": # No input file specified: creating empty track
            self.__track = mido.MidiTrack()
            self.__time_signature_msg = mido.MetaMessage('time_signature',
                                                 numerator=self.__time_signature.numerator,
                                                 denominator=self.__time_signature.denominator)
            self.__track.append(self.__time_signature_msg)
            self.__track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.__latest_tempo_bpm), time=0))

        else:   # Parsing file
            file = mido.MidiFile(file_path)
            self.__ticks_per_beat = file.ticks_per_beat
            print(f'Reading MIDI file: {file_path}')
            print(f'Ticks / Beat = {self.__ticks_per_beat}')
            print('Available tracks:')
            for i, track in enumerate(file.tracks):
                print('Track {}: {}'.format(i, track.name))
            self.__track = file.tracks[track_idx]
            print(f'Selected track: {self.__track.name}')
            for msg in self.__track:
                if msg.type == 'set_tempo':
                    self.__latest_tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':  # Not actually used to determine the length of notes.
                    self.__time_signature.numerator = msg.numerator
                    self.__time_signature.denominator = msg.denominator
                elif msg.type == 'note_on' and msg.velocity > 0:
                    self.__latest_velocity = msg.velocity
                    self.__current_rest_length = 1

    def collect_notes(self):
        notes: list[music.Note] = []
        playing_notes: set[int] = set()
        for msg in self.__track:
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note not in playing_notes:
                if msg.time > 1:    # Insert rest (time = 1 is not a rest)
                    notes.append(music.Note(
                        note=0,
                        length_beats=(msg.time - 1) / self.__ticks_per_beat,
                        is_rest=True
                    ))
                playing_notes.add(msg.note)
            if (msg.type == 'note_on' and msg.velocity == 0) or msg.type == 'note_off' and msg.note in playing_notes:
                playing_notes.discard(msg.note)
                notes.append(music.Note(
                    note=int(msg.note - 60),
                    length_beats=(msg.time + 1) / self.__ticks_per_beat
                ))
        return notes

    def append_rest(self, length_beats):
        self.__current_rest_length += length_beats * self.__ticks_per_beat

    def append_note(self, note: music.Note):
        if note.is_rest:
            self.append_rest(note.length_beats)
        else:
            midi_note = 60 + note.note
            self.__track.append(mido.Message('note_on', note=midi_note, velocity=self.__latest_velocity, time=self.__current_rest_length))
            self.__track.append(mido.Message('note_on', note=midi_note, velocity=0, time=note.length_beats * self.__ticks_per_beat - 1))
            self.__current_rest_length = 1

    def append_tempo_change(self, new_tempo_bpm: int):
        self.__track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(new_tempo_bpm)))

    def change_velocity(self, velocity: int):
        self.__latest_velocity = velocity

    def change_time_singature(self, time_signature: music.TimeSignature):
        self.__time_signature = time_signature
        self.__time_signature_msg.numerator = self.__time_signature.numerator
        self.__time_signature_msg.denominator = self.__time_signature.denominator

    def play(self, speed_multiplier = 1):
        outputs = mido.get_output_names()
        print("Available outputs:")
        for output in outputs:
            print(output)
        print("")
        output = mido.open_output(outputs[0])
        clocks_per_click = 24
        tempo_bpm = 120
        for msg in self.__track:
            if not type(msg) == mido.MetaMessage:
                time.sleep(msg.time / self.__ticks_per_beat / tempo_bpm * 60 / speed_multiplier)
                output.send(msg)
            elif msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == 'time_signature':
                clocks_per_click = msg.clocks_per_click
        output.close()

    def save(self, file_path: pathlib.Path):
        file = mido.MidiFile()
        file.ticks_per_beat = self.__ticks_per_beat
        file.tracks.append(self.__track)
        file.save(file_path)