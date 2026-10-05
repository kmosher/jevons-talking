// Jev answers a question one menu pick at a time, like a word-prediction keyboard:
// each step offers predicted next words alongside letters that narrow
// the predictions, plus punctuation, backspace and SPEAK. Jev picks one option per call.
//
// Word prediction is a bigram model over WordNet glosses and their example sentences,
// with some slots reserved for the most common words so a function word like "is"
// is always on offer even after a word with many followers. Words in blocklist.txt
// are never offered and can't be entered by spelling them out.
//
// Each call sees its last few actions, and an option it later backspaced over is
// removed from that menu, so a stateless Jev can't loop on the same dead end.
//
// Usage: npm run talk -- [--q="What is a black hole?"] [--max-steps=100] [--words=150] [--common=30] [--dry-run]
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'
import { choice, TypeSafeClient } from '@typesafe-ai/sdk'

export type Step = { menu: string[]; pick: string; confidence: number; probabilities: Record<string, number> }
export type Talk = { question: string; answer: string; finished: boolean; steps: Step[] }
type Options = { maxSteps?: number; words?: number; common?: number; dryRun?: boolean; log?: (line: string) => void }

// --- Blocklist ---------------------------------------------------------------
const blocked = new Set(
  readFileSync(new URL('blocklist.txt', import.meta.url), 'utf8')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#')),
)
export const isBlocked = (word: string) => blocked.has(word) || (word.endsWith('s') && blocked.has(word.slice(0, -1)))
export const containsBlocked = (text: string) => (text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) ?? []).some(isBlocked)

// --- Predictor ---------------------------------------------------------------
const DICT: string = createRequire(import.meta.url)('wordnet-db').path
const unigram = new Map<string, number>()
const bigram = new Map<string, Map<string, number>>()
const START = '<s>'
for (const pos of ['noun', 'verb', 'adj', 'adv']) {
  for (const line of readFileSync(join(DICT, `data.${pos}`), 'utf8').split('\n')) {
    const gloss = line.split(' | ')[1]
    if (!gloss) continue
    for (const sentence of gloss.toLowerCase().split(/[;"]/)) {
      const words = sentence.match(/[a-z]+(?:'[a-z]+)?/g)?.filter((w) => !isBlocked(w))
      if (!words) continue
      let prev = START
      for (const w of words) {
        unigram.set(w, (unigram.get(w) ?? 0) + 1)
        const m = bigram.get(prev) ?? new Map<string, number>()
        m.set(w, (m.get(w) ?? 0) + 1)
        bigram.set(prev, m)
        prev = w
      }
    }
  }
}
const byFrequency = [...unigram.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w)

// The `common` most frequent words matching the typed prefix, then bigram
// continuations of `prev`, then unigram backoff to fill `n` slots.
function predict(prev: string, prefix: string, n: number, common: number): string[] {
  const out = new Set<string>()
  const follow = [...(bigram.get(prev)?.entries() ?? [])].sort((a, b) => b[1] - a[1]).map(([w]) => w)
  const take = (source: string[], limit: number) => {
    for (const w of source) {
      if (out.size >= limit) break
      if (w.startsWith(prefix)) out.add(w)
    }
  }
  take(byFrequency, common)
  take(follow, n)
  take(byFrequency, n)
  return [...out]
}

// --- Menu loop ---------------------------------------------------------------
const instructions = `You are composing a spoken answer to a question using an assistive
communication menu. You cannot type freely: each turn you pick exactly one
menu option. Options are:
- "word: X" — append the predicted word X to your sentence.
- "letter: X" — type a letter to narrow the word predictions to words starting with what you've typed
  (use this when the word you want is not among the predictions). Digits work the same way.
- punctuation, "backspace" (undo the last letter typed, or else the last word), and "SPEAK" (finish:
  your text is spoken aloud as your final answer).
recent_actions lists your last few picks. Options you already backspaced over from the current text
are not offered again: if you're stuck, backspace further and rephrase.
Aim for a short, correct answer of one or two sentences, then pick SPEAK.`
const HISTORY = 10
let client: TypeSafeClient | undefined

export async function talk(question: string, opts: Options = {}): Promise<Talk> {
  const { maxSteps = 100, words: nWords = 150, common = 30, dryRun = false, log = () => {} } = opts
  client ??= new TypeSafeClient()
  const words: string[] = []
  let prefix = ''
  // Forward moves, so a backspace knows what it undid and from which state.
  const undo: { from: string; option: string }[] = []
  const rejected = new Map<string, Set<string>>()
  const steps: Step[] = []
  const key = () => `${words.join(' ')}|${prefix}`
  const text = () => words.join(' ').replace(/ ([.,?])/g, '$1')

  for (let step = 0; step < maxSteps; step++) {
    const prev = words.length && !/[.,?]/.test(words.at(-1)!) ? words.at(-1)! : START
    const menu: Record<string, string> = {}
    for (const w of predict(prev, prefix, nWords, common)) menu[`word: ${w}`] = `append "${w}"`
    const letters = new Set(byFrequency.filter((w) => w.startsWith(prefix) && w.length > prefix.length).map((w) => w[prefix.length]))
    if (/^\d*$/.test(prefix)) for (const d of '0123456789') letters.add(d)
    if (prefix && !isBlocked(prefix)) menu[`word: ${prefix}`] ??= `enter "${prefix}" as typed`
    for (const l of [...letters].sort()) menu[`letter: ${l}`] = `narrow predictions to words starting "${prefix + l}"`
    if (!prefix) for (const p of ['.', ',', '?']) menu[p] = `append "${p}"`
    if (prefix || words.length) menu.backspace = prefix ? `delete the typed letter "${prefix.at(-1)}"` : `delete "${words.at(-1)}"`
    if (words.length && !prefix) menu.SPEAK = 'finish and speak the text aloud'
    for (const o of rejected.get(key()) ?? []) delete menu[o]

    const recent = steps.slice(-HISTORY).map((s) => s.pick)
    const state = { instructions, question, recent_actions: recent.length ? recent : '(none)', text_so_far: text() || '(nothing yet)', letters_typed: prefix || '(none)' }
    const questions = { next: choice('Which menu option do you pick next?', menu) }
    if (dryRun) {
      console.log(JSON.stringify({ state, questions }, null, 2))
      break
    }
    const r = await client.systemOne({ state, questions })
    const a = r.answers.next as { choice: string; confidence: number; probabilities: Record<string, number> }
    steps.push({ menu: Object.keys(menu), pick: a.choice, confidence: a.confidence, probabilities: a.probabilities })
    log(`${String(step + 1).padStart(2)} ${a.choice.padEnd(16)} conf=${a.confidence.toFixed(2)}  | ${text()}${prefix ? ' ' + prefix + '…' : ''}`)

    if (a.choice === 'SPEAK') break
    if (a.choice === 'backspace') {
      const last = undo.pop()
      if (last) rejected.set(last.from, (rejected.get(last.from) ?? new Set()).add(last.option))
      if (prefix) prefix = prefix.slice(0, -1)
      else words.pop()
      continue
    }
    undo.push({ from: key(), option: a.choice })
    if (a.choice.startsWith('letter: ')) prefix += a.choice.slice(8)
    else if (a.choice.startsWith('word: ')) {
      words.push(a.choice.slice(6))
      prefix = ''
    } else words.push(a.choice)
  }
  return { question, answer: text(), finished: steps.at(-1)?.pick === 'SPEAK', steps }
}

export function save(t: Talk): string {
  mkdirSync('transcripts', { recursive: true })
  const out = `transcripts/${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  writeFileSync(out, JSON.stringify(t, null, 2))
  return out
}

export const meanConfidence = (t: Talk) => t.steps.reduce((x, s) => x + s.confidence, 0) / (t.steps.length || 1)

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const dryRun = args.includes('--dry-run')
  const t = await talk(flag('q') ?? 'What is a black hole?', {
    maxSteps: Number(flag('max-steps') ?? 100),
    words: Number(flag('words') ?? 150),
    common: Number(flag('common') ?? 30),
    dryRun,
    log: (l) => console.error(l),
  })
  if (!dryRun) {
    console.log(`\nQ: ${t.question}\nA: ${t.answer}${t.finished ? '' : '  [step cap hit]'}`)
    const confs = t.steps.map((s) => s.confidence)
    console.log(`${t.steps.length} calls, mean confidence ${meanConfidence(t).toFixed(2)}, min ${Math.min(...confs).toFixed(2)}`)
    console.log(`saved ${save(t)}`)
  }
}
