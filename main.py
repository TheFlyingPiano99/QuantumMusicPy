# Markov Music Py project by Zoltán Simon 2024/25

import quantum_music.quantum_model as quantum_model
import quantum_music.midi_layer as midi_layer
import quantum_music.music_layer as music
import math
import time


def main():
    print('Hi, Markov Music!')
    midi = midi_layer.MidiTrack("resources/midi/Boci boci tarka.mid")
    #midi = midi_layer.MidiTrack()

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

    start_time = time.time()
    model = quantum_model.QuantumModel(look_back_steps=1, look_back_note_length=False)
    notes = midi.collect_notes()
    print('\nNotes of the song:')
    for note in notes:
        print(note)
    print('')

    model.build_operator_from_notes(notes)
    #model.build_descending_chromatic_scale_operator(phase=math.pi / 8.0)
    #model.build_bidirectional_chromatic_scale_operator()
    model.init_measurement_base()
    model.init_classical_state([model.placeholder_rest(), music.Note(note=0, length_beats=1)], phase=0.0)
    #model.init_eigen_state(20)
    #model.build_bidirectional_major_scale_operator(phase=0.0)
    model.test_indexing()
    model.test_measurement_base()
    model.test_density_matrix()

    midi.next_note_index = len(notes)   # Skip the playback of the original song
    midi.play(speed_multiplier=2)
    print('\nGenerating more notes:')
    for i in range(1000):
        harmony = model.measure_state(
            max_velocity=80,
            superposition_voices=4,
            collapse_state=False,
            fuzzy_measurement=False
        )
        for note in harmony:
            print(note)
        #if i % 500 == 0:
            #print('Invert operator')
            #model.invert_evolution_opearotor()
        midi.append_harmony(harmony)
        print('')
        model.evolve_state(1)
    end_time = time.time()
    duration = end_time - start_time
    print(f'Calculation took {duration} seconds.')
    midi.save("resources/midi/generated scale.mid")


if __name__ == '__main__':
    main()
