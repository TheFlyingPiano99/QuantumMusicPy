# Markov Music Py project by Zoltán Simon 2024/25
import pathlib

import quantum_music.quantum_model as quantum_model
import quantum_music.midi_layer as midi_layer
import quantum_music.music_layer as music
import math
import time


def main():
    print('Hi, Markov Music!')

    # MIDI data:
    #midi = midi_layer.MidiTrack(pathlib.Path("resources/midi/Régi stílusú palóc népdalok.mid"))
    midi = midi_layer.MidiTrack()
    notes = midi.collect_notes()

    # Evolution operator:
    model = quantum_model.QuantumModel(look_back_steps=2, look_back_note_length=False)
    model.load_evolution_operator(pathlib.Path('saves/operators/hungarian_folk_song_op.npy'))
    #model.build_operator_from_notes(notes)
    #model.serialise_evolution_operator(pathlib.Path('saves/operators/hungarian_folk_song_op.npy'))

    # Measurement base:
    model.init_measurement_base()
    model.transfer_measurement_base_to_gpu()

    # Initial state:
    print('Initial state')
    phase = 0.0
    model.init_superposition_state(
        [
            #[
                #music.Note(note=0, length_beats=0.5, is_rest=False),
                #music.Note(note=10, length_beats=0.5, is_rest=False),
                #music.Note(note=9, length_beats=0.5, is_rest=False)
            #],
            [
                music.Note(note=0, length_beats=0.5, is_rest=False),
                music.Note(note=2, length_beats=0.5, is_rest=False),
                music.Note(note=0, length_beats=0.5, is_rest=False)
            ],
        ],
        [
            #math.sqrt(4 / 8) * (math.cos(phase) + 1j * math.sin(phase)),
            math.sqrt(8 / 8) * (math.cos(phase * 2) + 1j * math.sin(phase * 2)),
        ]
    )

    # Simulation loop:
    midi.next_note_index = len(notes)   # Skip the playback of the original song
    playback_thread = midi.play(speed_multiplier=1)       # Start playback on a different thread
    print('\nGenerating sequence:')
    start_time = time.time()
    total_chord_count = 129
    for i in range(total_chord_count):
        harmony = model.measure_state(
            max_velocity=80,
            superposition_voices=2,
            collapse_state=False,
            fuzzy_measurement=False
        )
        for note in harmony:
            print(note)
        if i > 0 and i % (total_chord_count // 2) == 0:
            print('--- Invert operator -------------------------------------------------')
            model.invert_evolution_opearotor()
        midi.append_harmony(harmony)
        model.evolve_state(1)
    end_time = time.time()
    duration = end_time - start_time
    print(f'Calculation took {duration} seconds.')

    # Output serialisation:
    file_name = 'folksong_variations.mid'
    print(f'Saving to file: {file_name}')
    midi.save(pathlib.Path("saves/midi/" + file_name))
    playback_thread.join()      # Don't exit until the playback has finished


if __name__ == '__main__':
    main()
