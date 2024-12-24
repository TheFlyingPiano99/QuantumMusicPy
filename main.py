# Markov Music Py project by Zoltán Simon 2024/25
import math

import quantum_music.quantum_model as quantum_model
import quantum_music.midi_layer as midi_layer
import quantum_music.music_layer as music

def main():
    print('Hi, Markov Music!')
    midi = midi_layer.MidiTrack("resources/midi/Boci boci tarka.mid")

    # midi.append_tempo_change(120)
    # midi.change_velocity(60)
    # midi.append_note(music.Note(0, 1))
    # midi.append_note(music.Note(-1, 1))
    # midi.append_note(music.Note(0, 1))
    # midi.append_note(music.Note(-1, 1))
    # midi.append_rest(4)
    # midi.change_velocity(40)
    # midi.append_note(music.Note(0, 4))
    # midi.change_velocity(80)

    model = quantum_model.QuantumModel(look_back_steps=1, look_back_note_length=False)
    notes = midi.collect_notes()
    print('\nNotes of the song:')
    for note in notes:
        print(note)
    print('')

    #model.build_operator_from_notes(notes)
    model.build_chromatic_scale_operator()
    model.init_projective_measurement_base()
    model.init_state_as_base_state(0)

    print('\nGenerating more notes:')
    for i in range(100):
        model.evolve_state(1)
        harmony = model.measure_state(max_velocity=80, superposition_voices=4, collapse_state=False, fuzzy_measurement=True)
        print('')
        for note in harmony:
            print(note)
        midi.append_harmony(harmony)

    print("\n\nPlayback:")
    midi.play(speed_multiplier=2)
    midi.save("resources/midi/Hungarian children's songs with generated variations.mid")


if __name__ == '__main__':
    main()
