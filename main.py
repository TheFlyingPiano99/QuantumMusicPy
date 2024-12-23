# Markov Music Py project by Zoltán Simon 2024/25
import math

import quantum_music.quantum_model as quantum_model
import quantum_music.midi_layer as midi_layer
import quantum_music.music_layer as music

def main():
    print('Hi, Markov Music!')
    midi = midi_layer.MidiTrack("resources/midi/Boci boci tarka (tempo change).mid")

    midi.append_tempo_change(120)
    midi.change_velocity(60)
    midi.append_note(music.Note(0, 1))
    midi.append_note(music.Note(-1, 1))
    midi.append_note(music.Note(0, 1))
    midi.append_note(music.Note(-1, 1))
    midi.append_rest(4)
    midi.change_velocity(40)
    midi.append_note(music.Note(0, 4))
    midi.change_velocity(80)

    model = quantum_model.QuantumModel(look_back_steps=1, look_back_note_length=False)
    notes = midi.collect_notes()
    print('Notes of the song:')
    for note in notes:
        print(note)

    model.build_operator_from_notes(notes)
    model.init_classical_state(music.Note(0, 2), phase=math.pi / 2)
    model.init_projective_measurement_operators()

    print('Generating more notes:')
    for i in range(100):
        model.evolve_state()
        notes = model.measure_state(collapse_state=False, probabilistic=True)
        for note in notes:
            print(note)
            midi.append_note(note)

    print("\nPlayback:")
    midi.play(speed_multiplier=2)
    #midi.save("resources/midi/Boci boci tarka 2.mid")


if __name__ == '__main__':
    main()
