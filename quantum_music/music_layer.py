from typing import Union


note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
note_lengths = {0.25: 'sixteenth',
                0.375: 'dotted sixteenth',
                0.5: 'eight',
                0.75: 'dotted eight',
                1: 'quarter',
                1.5: 'dotted quarter',
                2: 'half',
                3: 'dotted half',
                4: 'whole',
                6: 'dotted whole'}

class TimeSignature:
    numerator: int = 4
    denominator: int = 4

    def __str__(self):
        return f'{self.numerator} / {self.denominator}'

class Note:
    note: int   # C is 0. Steps in halftones
    length_beats: Union[int, float]
    is_rest: bool

    def __init__(self, note, length_beats, is_rest: bool = False):
        self.note = note
        self.length_beats = length_beats
        self.is_rest = is_rest

    def __str__(self):
        if self.length_beats in note_lengths:
            length = note_lengths[self.length_beats]
        else:
            length = self.length_beats
        return f'{note_names[self.note % 12]} {length} note' if not self.is_rest else f'{length} rest'
