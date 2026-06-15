# Sequence Models Week 1, Assignment 3 (Jazz Improvisation with LSTM)

This assignment notebook was originally developed for older versions of TensorFlow, Keras, and music21. Running it locally with newer library versions requires the following setup.

## Required Package Versions

### music21

Install `music21==6.5.0`:

```bash
pip install music21==6.5.0
```

Newer versions of `music21` changed how `getElementsByClass()` returns results, which causes the following error in `preprocess.py` when parsing the MIDI file:

```python
ValueError
...
File preprocess.py:29, in __parse_midi(data_fn)
     27 # Get melody part, compress into single voice.
     28 melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.
---> 29 melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
```

Installing `music21==6.5.0` resolves this without modifying `preprocess.py`.
