// Jev answers a question one menu pick at a time, like a word-prediction keyboard:
// each step offers predicted next words alongside letters that narrow
// the predictions, plus punctuation, backspace and SPEAK. Jev picks one option per call.
//
// Word prediction is a bigram model over WordNet glosses and their example sentences,
// with some slots reserved for the most common words so a function word like "is"
// is always on offer even after a word with many followers. Words in blocklist.txt
// are never offered and can't be entered by spelling them out.
//
// With confirmSpeak, picking SPEAK prompts a yes/no question asking whether that's Jev's final
// answer; a "no" withdraws SPEAK at that point and Jev keeps going, up to maxRejections times.
//
// Each call sees its last few actions, and an option it later backspaced over is
// removed from that menu, so a stateless Jev can't loop on the same dead end.
//
// Usage: npm run talk -- [--q="What is a black hole?"] [--max-steps=100] [--words=150] [--common=30] [--dry-run]
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { join } from 'node:path'
import { choice, noul, TypeSafeClient } from '@typesafe-ai/sdk'

export type Step = { menu: string[]; pick: string; confidence: number; probabilities: Record<string, number>; final?: number }
export type Talk = { question: string; answer: string; finished: boolean; steps: Step[] }
export type Post = { author: string; text: string }
type Options = { phrases?: number; conversation?: Post[]; extraInstructions?: string; minWordsToSpeak?: number; confirmSpeak?: boolean; maxRejections?: number; maxSteps?: number; words?: number; common?: number; dryRun?: boolean; log?: (line: string) => void }

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
export const instructions = `You are composing a spoken answer to a question using an assistive
communication menu. You cannot type freely: each turn you pick exactly one
menu option. Options are:
- "word: X" — append the predicted word X to your sentence.
- "letter: X" — type a letter to narrow the word predictions to words starting with what you've typed
  (use this when the word you want is not among the predictions). Any letter or digit can be typed, so
  you can spell any word, then enter it as typed.
- punctuation, "backspace" (undo the last letter typed, or else the last word), and "SPEAK" (finish:
  your text is spoken aloud as your final answer).
recent_actions lists your last few picks. Options you already backspaced over from the current text
are not offered again: if you're stuck, backspace further and rephrase.
If conversation_so_far is present, the question is the latest message in that conversation; posts
by "you" are your own earlier replies, and "@you" in a message means it is addressed to you.
Aim for a short, correct answer of one or two sentences, then pick SPEAK.`
const HISTORY = 10
let client: TypeSafeClient | undefined
export const jev = () => (client ??= new TypeSafeClient())

// One draft in progress: its text, the letters typed toward the next word, and what it has rejected.
export type Branch = {
  words: string[]
  prefix: string
  // Forward moves, so a backspace knows what it undid and from which state.
  undo: { from: string; option: string }[]
  rejected: Map<string, Set<string>>
  steps: Step[]
  rejections: number
  score: number
}
// Bigram followers that make a phrase: at least this share of what follows the word, and seen this often.
const PHRASE_SHARE = 0.15
const PHRASE_MIN = 5
const followerCache = new Map<string, string | null>()
function topFollower(w: string): string | null {
  if (!followerCache.has(w)) {
    const m = bigram.get(w)
    let best: string | null = null
    if (m) {
      const total = [...m.values()].reduce((a, b) => a + b, 0)
      const [top, n] = [...m.entries()].sort((a, b) => b[1] - a[1])[0]
      if (n >= PHRASE_MIN && n / total >= PHRASE_SHARE) best = top
    }
    followerCache.set(w, best)
  }
  return followerCache.get(w)!
}

export type MenuOptions = { words: number; common: number; minWordsToSpeak: number; phrases?: number }

export const newBranch = (): Branch => ({ words: [], prefix: '', undo: [], rejected: new Map(), steps: [], rejections: 0, score: 1 })
export const cloneBranch = (b: Branch): Branch => ({
  ...b,
  words: [...b.words],
  undo: [...b.undo],
  rejected: new Map([...b.rejected].map(([k, v]) => [k, new Set(v)])),
  steps: [...b.steps],
})
export const branchKey = (b: Branch) => `${b.words.join(' ')}|${b.prefix}`
export const branchText = (b: Branch) => b.words.join(' ').replace(/ ([.,?])/g, '$1')
const reject = (b: Branch, key: string, option: string) => b.rejected.set(key, (b.rejected.get(key) ?? new Set()).add(option))

export function menuFor(b: Branch, o: MenuOptions): Record<string, string> {
  const { words, prefix } = b
  // A phrase is stored as one entry; predictions follow its last word.
  const prev = words.length && !/[.,?]/.test(words.at(-1)!) ? words.at(-1)!.split(' ').at(-1)! : START
  const menu: Record<string, string> = {}
  const predicted = predict(prev, prefix, o.words, o.common)
  for (const w of predicted) menu[`word: ${w}`] = `append "${w}"`
  // Two-word phrases: a predicted word with its most likely follower, when that pairing is strong.
  if (o.phrases) {
    let added = 0
    for (const w of predicted) {
      if (added >= o.phrases) break
      const next = topFollower(w)
      if (next) {
        menu[`word: ${w} ${next}`] = `append "${w} ${next}"`
        added++
      }
    }
  }
  // Any letter or digit, like a real keyboard, so Jev can spell words the predictor doesn't know.
  const letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
  if (prefix && !isBlocked(prefix)) menu[`word: ${prefix}`] ??= `enter "${prefix}" as typed`
  for (const l of letters) menu[`letter: ${l}`] = `narrow predictions to words starting "${prefix + l}"`
  // No punctuation straight after punctuation, so Jev can't stall on "no,.....".
  if (!prefix && !/^[.,?]$/.test(words.at(-1) ?? '')) for (const p of ['.', ',', '?']) menu[p] = `append "${p}"`
  if (prefix || words.length) menu.backspace = prefix ? `delete the typed letter "${prefix.at(-1)}"` : `delete "${words.at(-1)}"`
  if (words.filter((w) => /\w/.test(w)).length >= o.minWordsToSpeak && !prefix) menu.SPEAK = 'finish and speak the text aloud'
  for (const opt of b.rejected.get(branchKey(b)) ?? []) delete menu[opt]
  return menu
}

// Applies a non-SPEAK pick to the branch.
export function applyPick(b: Branch, pick: string) {
  if (pick === 'backspace') {
    const last = b.undo.pop()
    if (last) reject(b, last.from, last.option)
    if (b.prefix) b.prefix = b.prefix.slice(0, -1)
    else b.words.pop()
    return
  }
  b.undo.push({ from: branchKey(b), option: pick })
  if (pick.startsWith('letter: ')) b.prefix += pick.slice(8)
  else if (pick.startsWith('word: ')) {
    b.words.push(pick.slice(6))
    b.prefix = ''
  } else b.words.push(pick)
}

export const recentActions = (b: Branch) => {
  const recent = b.steps.slice(-HISTORY).map((s) => s.pick)
  return recent.length ? recent : '(none)'
}

export async function talk(question: string, opts: Options = {}): Promise<Talk> {
  const { phrases = 0, conversation, extraInstructions, minWordsToSpeak = 1, confirmSpeak = false, maxRejections = 2, maxSteps = 100, words: nWords = 150, common = 30, dryRun = false, log = () => {} } = opts
  client ??= new TypeSafeClient()
  const b = newBranch()

  for (let step = 0; step < maxSteps; step++) {
    const menu = menuFor(b, { words: nWords, common, minWordsToSpeak, phrases })
    const state = { instructions: extraInstructions ? `${instructions}\n${extraInstructions}` : instructions, ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, recent_actions: recentActions(b), text_so_far: branchText(b) || '(nothing yet)', letters_typed: b.prefix || '(none)' }
    const questions = { next: choice('Which menu option do you pick next?', menu) }
    if (dryRun) {
      console.log(JSON.stringify({ state, questions }, null, 2))
      break
    }
    const r = await client.systemOne({ state, questions })
    const a = r.answers.next as { choice: string; confidence: number; probabilities: Record<string, number> }
    b.steps.push({ menu: Object.keys(menu), pick: a.choice, confidence: a.confidence, probabilities: a.probabilities })
    log(`${String(step + 1).padStart(2)} ${a.choice.padEnd(16)} conf=${a.confidence.toFixed(2)}  | ${branchText(b)}${b.prefix ? ' ' + b.prefix + '…' : ''}`)

    if (a.choice === 'SPEAK') {
      if (!confirmSpeak || b.rejections >= maxRejections) break
      // A second opinion on stopping: a "no" withdraws SPEAK from this point and Jev carries on.
      const c = await client.systemOne({ state, questions: { final: noul('Is the text so far your final answer?') } })
      const final = (c.answers.final as { noul: number }).noul
      b.steps.at(-1)!.final = final
      log(`   final answer? ${final.toFixed(2)}`)
      if (final >= 0.5) break
      reject(b, branchKey(b), 'SPEAK')
      b.rejections++
      // With no checks left, SPEAK is open again everywhere; leaving old blocks in place sent
      // Jev on detours, deleting words just to reach a text it was allowed to finish.
      if (b.rejections >= maxRejections) for (const set of b.rejected.values()) set.delete('SPEAK')
      continue
    }
    applyPick(b, a.choice)
  }
  const last = b.steps.at(-1)
  return { question, answer: branchText(b), finished: last?.pick === 'SPEAK' && (last.final ?? 1) >= 0.5, steps: b.steps }
}

export function save(t: Talk): string {
  mkdirSync('transcripts', { recursive: true })
  const out = `transcripts/${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  writeFileSync(out, JSON.stringify(t, null, 2))
  return out
}

// A bare yes or no ("no", "no.", "no. no"): allowed, but it has to clear a higher bar to win.
export const isBare = (text: string) => /^\s*((yes|no)[\s.,?!]*)+$/i.test(text)

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
