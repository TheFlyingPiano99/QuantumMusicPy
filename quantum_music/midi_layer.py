import mido     # Source: https://mido.readthedocs.io/en/stable/index.html
import pathlib
import time
import quantum_music.music_layer as music
from threading import Thread


class MidiTrack:
    __ticks_per_beat: int
    __latest_tempo_bpm: int
    __latest_velocity: int
    __track: mido.MidiTrack
    __time_signature: music.TimeSignature
    __current_rest_length: int
    __next_note_index: int

    def __init__(self, file_path: pathlib.Path = None, track_idx=0):
        self.__ticks_per_beat = 480
        self.__latest_tempo_bpm = 120
        self.__latest_velocity = 64
        self.__current_rest_length = 0
        self.__next_note_index = 0
        self.__time_signature = music.TimeSignature()
        self.__time_signature_msg: mido.MetaMessage

        if file_path is None:   # No input file specified: creating empty track
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
            self.__track.append(mido.Message(
                type='note_on',
                note=midi_note,
                velocity=int(note.velocity),
                time=int(self.__current_rest_length)
            ))
            self.__track.append(mido.Message(
                type='note_on',
                note=midi_note,
                velocity=0,
                time=int(note.length_beats * self.__ticks_per_beat - 1)
            ))
            self.__current_rest_length = 1

    def append_harmony(self, harmony: list[music.Note]):
        is_full_rest = True
        loudest_length = 0
        loudest_velocity = 0
        # Append note-starts:
        for note in harmony:
            if note.velocity > loudest_velocity:
                loudest_velocity = note.velocity
                loudest_length = note.length_beats
            if not note.is_rest:
                is_full_rest = False
                midi_note = 60 + note.note
                self.__track.append(mido.Message(
                    type='note_on',
                    note=midi_note,
                    velocity=note.velocity,
                    time=int(self.__current_rest_length)
                ))
                self.__current_rest_length = 0
        # Append note-ends:
        is_first_end = True
        for note in harmony:
            if not note.is_rest:
                midi_note = 60 + note.note
                self.__track.append(mido.Message(
                    type='note_on',
                    note=midi_note,
                    velocity=0,
                    time=int(loudest_length * self.__ticks_per_beat - 1) if is_first_end else 0
                ))
                is_first_end = False
                self.__current_rest_length = 1
        if is_full_rest:
            self.append_rest(loudest_length)

    def append_tempo_change(self, new_tempo_bpm: int):
        self.__track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(new_tempo_bpm)))

    def change_velocity(self, velocity: int):
        self.__latest_velocity = velocity

    def change_time_singature(self, time_signature: music.TimeSignature):
        self.__time_signature = time_signature
        self.__time_signature_msg.numerator = self.__time_signature.numerator
        self.__time_signature_msg.denominator = self.__time_signature.denominator

    def __play_routine(self, speed_multiplier: float = 1):
        time.sleep(3)
        print("\n\nPlayback:")
        outputs = mido.get_output_names()
        print("Available outputs:")
        for output in outputs:
            print(output)
        print("")
        output = mido.open_output(outputs[0])
        clocks_per_click = 24
        tempo_bpm = 120
        note_index = 0
        arrived_to_start = False
        for msg in self.__track:
            if not type(msg) == mido.MetaMessage:
                if msg.type == 'note_on' and msg.velocity > 0:  # This condition is used to control next_note_index
                    note_index += 1
                    if note_index < self.__next_note_index:     # Skip note playing if not at the next_note_index
                        continue
                    else:
                        arrived_to_start = True
                        self.__next_note_index += 1
                if arrived_to_start:    # Only sleep if already playing notes
                    time.sleep(msg.time / self.__ticks_per_beat / tempo_bpm * 60 / speed_multiplier)
                output.send(msg)
            elif msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == 'time_signature':
                clocks_per_click = msg.clocks_per_click
        self.__next_note_index = 0  # Reset index
        time.sleep(2)
        output.close()
        print('Finished playback.')


    def play(self, speed_multiplier: float = 1):
        thread = Thread(target=self.__play_routine, args=(speed_multiplier, ))
        thread.start()
        return thread

    def save(self, file_path: pathlib.Path):
        file = mido.MidiFile()
        file.ticks_per_beat = self.__ticks_per_beat
        file.tracks.append(self.__track)
        file.save(file_path)


    def get_next_note_index(self):
        return self.__next_note_index

    def set_next_note_index(self, value):
        self.__next_note_index = value

    next_note_index = property(get_next_note_index, set_next_note_index)
