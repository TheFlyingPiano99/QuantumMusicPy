# Markov Music Py project by Zoltán Simon 2024/25

import markov_music.markov_model as markov_model
import markov_music.midi_layer as midi_layer


def main():
    print('Hi, Markov Music!')
    midi = midi_layer.MidiData("resources/midi/Boci boci tarka.mid")
    model = markov_model.MarkovModel()


if __name__ == '__main__':
    main()
