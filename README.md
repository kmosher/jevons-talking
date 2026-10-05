# jevons-talking

[Jev](https://typesafe.ai) is a decision model: it can't write text, only pick from options.
This gives it a voice anyway, borrowing the idea from the word-prediction keyboards used in
assistive communication. Each turn, Jev sees a menu of
predicted next words, letters to narrow the predictions, punctuation, backspace and SPEAK, and
picks one. Code does everything else.

```
Q: What is the answer to life, the universe, and everything?

 1 word: the        conf=0.43  |
 2 backspace        conf=0.23  | the
 3 word: it         conf=0.18  |
 4 word: is         conf=0.45  | it
 5 letter: 4        conf=0.18  | it is
 6 letter: 2        conf=0.86  | it is 4…
 7 word: 42         conf=0.97  | it is 42…
 8 SPEAK            conf=0.80  | it is 42
```

A few more, unedited:

| Question | Jev's answer | Picks |
|---|---|---|
| What is a black hole? | black hole is an object that has a strong gravity | 15 |
| What is the capital of France? | the capital of france is paris | 21 |
| How do vaccines work? | vaccine is an instrument used against disease | 10 |
| What would you like to say to the world? | i want to do something that is responsible | 9 |

## How it works

- **Prediction** is a deliberately dumb bigram model over [WordNet](https://wordnet.princeton.edu/)'s
  glosses and example sentences: no language model is doing the talking. Each menu offers 150
  words — the 30 most common words, then the usual followers of the previous word. Typing a
  letter narrows the list to words with that prefix, which is how Jev reaches a word the
  predictor didn't think of.
- **Memory.** Every call is independent, so the state carries Jev's last 10 picks, and an option
  it backspaced over is dropped from that menu. Without this, Jev oscillates — adding a word,
  deleting it, adding it again — until it runs out of steps.
- **Confidence.** Each transcript in `transcripts/` records every menu, pick, confidence and the
  full probability vector.

## Running it

Needs Node 22+ and a TypeSafe API key in `TYPESAFE_API_KEY` (on macOS it's also read from the
login keychain, service `typesafe-jev`).

```sh
npm install
npm run talk -- --q="Why is the sky blue?"
```

Each pick is one API call; a run stops at SPEAK or after `--max-steps` (default 100).
`--dry-run` prints the first request without calling the API.
