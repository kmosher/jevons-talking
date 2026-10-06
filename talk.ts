// Jev answers a question one menu pick at a time, like a word-prediction keyboard:
// each step offers predicted next words alongside letters that narrow
// the predictions, plus punctuation, backspace and SPEAK. Jev picks one option per call.
//
// Word prediction is a bigram model over WordNet glosses and their example sentences (plus,
// with JEV_CHAT_WORDS=1, the conversational phrases in conversation.txt),
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
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { gunzipSync } from 'node:zlib'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { choice, noul, TypeSafeClient } from '@typesafe-ai/sdk'
import { chunkOf, dasherMenu } from './dasher.ts'

export type Step = { menu: string[]; pick: string; confidence: number; probabilities: Record<string, number>; final?: number }
export type Talk = { question: string; answer: string; finished: boolean; steps: Step[] }
export type Post = { author: string; text: string }
type Options = { keyboard?: Keyboard; mode?: Mode; phrases?: number; conversation?: Post[]; extraInstructions?: string; prefill?: string[]; extraWords?: string[]; minWordsToSpeak?: number; confirmSpeak?: boolean; maxRejections?: number; maxSteps?: number; words?: number; common?: number; dryRun?: boolean; log?: (line: string) => void }

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
function count(sentence: string, weight = 1) {
  const words = sentence.match(/[a-z]+(?:'[a-z]+)?/g)?.filter((w) => !isBlocked(w))
  if (!words) return
  let prev = START
  for (const w of words) {
    unigram.set(w, (unigram.get(w) ?? 0) + weight)
    const m = bigram.get(prev) ?? new Map<string, number>()
    m.set(w, (m.get(w) ?? 0) + weight)
    bigram.set(prev, m)
    prev = w
  }
}
for (const pos of ['noun', 'verb', 'adj', 'adv']) {
  for (const line of readFileSync(join(DICT, `data.${pos}`), 'utf8').split('\n')) {
    const gloss = line.split(' | ')[1]
    if (!gloss) continue
    for (const sentence of gloss.toLowerCase().split(/[;"]/)) count(sentence)
  }
}
// A small hand-written list of conversational phrases, weighted so that "thanks", "lol" or
// "fair enough" can compete with dictionary prose in the predictions.
// Opt-in with JEV_CHAT_WORDS=1.
const CONVERSATION_WEIGHT = 500
if (process.env.JEV_CHAT_WORDS === '1')
  for (const line of readFileSync(new URL('conversation.txt', import.meta.url), 'utf8').split('\n')) {
    if (line.trim() && !line.startsWith('#')) count(line.toLowerCase(), CONVERSATION_WEIGHT)
  }
// Contractions in context go into every word-pair table (see contractions.txt).
const CONTRACTION_WEIGHT = 200
if (process.env.JEV_CONTRACTIONS !== 'off')
  for (const line of readFileSync(new URL('contractions.txt', import.meta.url), 'utf8').split('\n')) {
    if (line.trim() && !line.startsWith('#')) count(line.toLowerCase(), CONTRACTION_WEIGHT)
  }

// The "chat" table adds Peter Norvig's web-scale word-pair counts (count_2w.txt, from Google's
// Web 1T corpus; downloaded once to ~/.cache/jevons-talking, or JEV_NORVIG_PAIRS), scaled so its
// biggest pair matches WordNet's biggest. Opt-in with JEV_NORVIG=on; otherwise it's the base table.
const chatBigram = new Map<string, Map<string, number>>([...bigram].map(([k, m]) => [k, new Map(m)]))
const NORVIG_URL = 'https://norvig.com/ngrams/count_2w.txt'
const norvigPath = process.env.JEV_NORVIG_PAIRS ?? join(homedir(), '.cache', 'jevons-talking', 'count_2w.txt')
if (process.env.JEV_NORVIG === 'on') {
  if (!existsSync(norvigPath)) {
    mkdirSync(dirname(norvigPath), { recursive: true })
    writeFileSync(norvigPath, await (await fetch(NORVIG_URL)).text())
  }
  let maxWordNet = 0
  for (const m of bigram.values()) for (const n of m.values()) if (n > maxWordNet) maxWordNet = n
  const pairs: [string, string, number][] = []
  for (const line of readFileSync(norvigPath, 'utf8').split('\n')) {
    const [both, n] = line.split('\t')
    const [a, b] = (both ?? '').toLowerCase().split(' ')
    if (!b || !/^[a-z]+$/.test(a) || !/^[a-z]+$/.test(b) || isBlocked(a) || isBlocked(b)) continue
    // Norvig splits contractions at the apostrophe ("don t"); those fragments are dropped.
    if (/^(t|s|re|ll|ve|m|d)$/.test(b) || /^(t|s|re|ll|ve|m|d)$/.test(a)) continue
    pairs.push([a, b, Number(n)])
  }
  const scale = maxWordNet / pairs.reduce((max, p) => Math.max(max, p[2]), 0)
  for (const [a, b, n] of pairs) {
    const m = chatBigram.get(a) ?? new Map<string, number>()
    m.set(b, (m.get(b) ?? 0) + n * scale)
    chatBigram.set(a, m)
  }
}

// Words are ranked (for the common slots, backoff and completions) by Android's keyboard
// frequencies from AOSP LatinIME, in data/, with JEV_RANKING=aosp; by default, by WordNet counts.
const AOSP = process.env.JEV_AOSP_WORDLIST ?? fileURLToPath(new URL('data/aosp_en_US_wordlist.combined.gz', import.meta.url))
const byFrequency =
  process.env.JEV_RANKING === 'aosp' && existsSync(AOSP)
    ? (() => {
        const raw = readFileSync(AOSP)
        const text = AOSP.endsWith('.gz') ? gunzipSync(raw).toString('utf8') : raw.toString('utf8')
        const f = new Map<string, number>()
        for (const m of text.matchAll(/word=([^,]+),f=(\d+)/g)) {
          const w = m[1].toLowerCase()
          if (/^[a-z]+(?:'[a-z]+)?$/.test(w) && !isBlocked(w)) f.set(w, Math.max(f.get(w) ?? 0, Number(m[2])))
        }
        return [...f.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w)
      })()
    : [...unigram.entries()].sort((a, b) => b[1] - a[1]).map(([w]) => w)

// The `common` most frequent words matching the typed prefix, then bigram
// continuations of `prev`, then unigram backoff to fill `n` slots.
// Two keyboards. "words": predicted next words from the bigram model, plus letters to narrow
// them. "letters": no predictions until Jev types something, then the most frequent words
// starting with it; two-letter keys for common letter pairs speed up the typing.
export type Keyboard = 'words' | 'chat' | 'letters'
// What Jev is told about each keyboard when it chooses one; written to be neutral, so none sounds
// more like "really you" than the others.
export const KEYBOARDS: Record<Keyboard, string> = {
  words:
    'Dictionary keyboard. It predicts your next word from word pairs in dictionary definitions, so it is good at ' +
    'definitional, factual sentences ("a black hole is a region of space…") and quick to write with. Its phrasing ' +
    'is formal and it rarely offers casual or conversational words.',
  chat:
    'Conversational keyboard. It predicts your next word from word pairs in everyday web text as well as dictionary ' +
    'definitions, so casual replies ("you too", "fair enough") are easy to reach. It can drift into loose or ' +
    'ungrammatical phrasing on factual questions.',
  letters:
    'Spelling keyboard. It predicts nothing until you type: you type letters or common letter pairs and pick from ' +
    'words that start with what you typed. Each word takes a few more picks, so replies tend to be short and ' +
    'telegraphic, but any word you can spell is available.',
}
const DEFAULT_KEYBOARD: Keyboard = process.env.JEV_PREDICTOR === 'completion' ? 'letters' : 'words'
// JEV_PREDICTOR=dasher swaps the whole keyboard for the Dasher-style character model in dasher.ts.
const DASHER = process.env.JEV_PREDICTOR === 'dasher'
// The 40 most frequent two-letter combinations, by word frequency, for the letters keyboard.
const PAIRS: string[] = (() => {
  const counts = new Map<string, number>()
  for (const [w, c] of unigram) for (let i = 0; i + 1 < w.length; i++) if (/^[a-z]{2}$/.test(w.slice(i, i + 2))) counts.set(w.slice(i, i + 2), (counts.get(w.slice(i, i + 2)) ?? 0) + c)
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, Number(process.env.JEV_PAIRS ?? 40)).map(([p]) => p)
})()

// Words that fit between prev and next: scored by how often they follow prev and precede next,
// so a replacement suits both neighbours. Falls back to the plain prediction after prev.
let preceders: Map<string, Map<string, number>> | undefined
export function fillers(prev: string, next: string | undefined, n: number): string[] {
  if (!preceders) {
    preceders = new Map()
    for (const [a, follows] of bigram) for (const [b, c] of follows) {
      const m = preceders.get(b) ?? preceders.set(b, new Map()).get(b)!
      m.set(a, c)
    }
  }
  const after = bigram.get(prev) ?? new Map<string, number>()
  const before = next ? (preceders.get(next) ?? new Map<string, number>()) : new Map<string, number>()
  const score = new Map<string, number>()
  for (const [w, c] of after) score.set(w, Math.log1p(c) + (before.has(w) ? Math.log1p(before.get(w)!) + 2 : 0))
  for (const [w, c] of before) if (!score.has(w)) score.set(w, Math.log1p(c))
  return [...score].sort((a, b) => b[1] - a[1]).map(([w]) => w).filter((w) => !isBlocked(w)).slice(0, n)
}

export function predict(prev: string, prefix: string, n: number, common: number, keyboard: Keyboard = DEFAULT_KEYBOARD): string[] {
  if (keyboard === 'letters') return prefix ? byFrequency.filter((w) => w.startsWith(prefix)).slice(0, n) : []
  const out = new Set<string>()
  const follow = [...((keyboard === 'chat' ? chatBigram : bigram).get(prev)?.entries() ?? [])].sort((a, b) => b[1] - a[1]).map(([w]) => w)
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
// The two control keys are offered as <end-word> and <end-phrase>, shaped unlike any word so
// they can't be read as the words "space" or "speak" (JEV_KEYNAMES=classic restores those).
// Transcripts and the trace image still call them SPACE and SPEAK.
const CLASSIC = process.env.JEV_KEYNAMES === 'classic'
export const SPACE = CLASSIC ? 'SPACE' : '<end-word>'
export const SPEAK = CLASSIC ? 'SPEAK' : '<end-phrase>'
export const isSpeak = (pick: string) => pick === SPEAK || pick === 'SPEAK' || pick === '<end-phrase>'
export const isSpace = (pick: string) => pick === SPACE || pick === 'SPACE' || pick === '<end-word>'
export const instructions = `${process.env.JEV_NAME === 'off' ? '' : 'You are Jev, called JT on Bluesky. '}You are composing a spoken answer to a question using an assistive
communication menu. You cannot type freely: each turn you pick exactly one
menu option. Options are:
- "word: X" — append the predicted word X to your sentence.
- "letter: X" — type a letter to narrow the word predictions to words starting with what you've typed
  (use this when the word you want is not among the predictions). Any letter or digit can be typed, so
  you can spell any word, then enter it as typed.
- "key: X" — type the punctuation mark X: every mark on a keyboard, plus — and …, and any mark can be
  typed as often as you like.
- "${SPACE}" — end the word you're typing, exactly as typed (after letters, punctuation or both).
- "backspace" (undo the last character typed, or else the last word), and "${SPEAK}" (finish: your text is
  spoken aloud as your final answer).
recent_actions lists your last few picks. Options you already backspaced over from the current text
are not offered again: if you're stuck, backspace further and rephrase.
If conversation_so_far is present, the question is the latest message in that conversation; posts
by "you" are your own earlier replies, and "@you" in a message means it is addressed to you.`

// What kind of reply a message calls for, chosen by Jev before it writes anything. Each mode
// sets the last line of the instructions and the wording the judges rate drafts against.
export const MODES = {
  answer: {
    when: 'it asks a question that can be answered',
    instruction: `Aim for a short, correct answer of one or two sentences, then pick ${SPEAK}.`,
    judge: 'a good answer to the question',
  },
  comeback: {
    when: 'it is a remark, claim, joke, challenge or provocation rather than a real question',
    instruction: `The message is not really a question: reply with a short, witty comeback, then pick ${SPEAK}.`,
    judge: 'a good, witty reply to the message',
  },
  react: {
    when: 'it shares a link, image or post and wants your reaction',
    instruction: `React to what was shared with a short, opinionated sentence, then pick ${SPEAK}.`,
    judge: 'a good reaction to what was shared',
  },
  yesno: {
    when: 'it is a yes-or-no question',
    instruction: `Your verdict on the yes-or-no question is already typed. Justify it in a few words, then pick ${SPEAK}.`,
    judge: 'a good answer to the yes-or-no question, with a reason',
  },
  acknowledge: {
    when: 'it is thanks, praise, a greeting or a goodbye',
    instruction: `Reply graciously in a few words, then pick ${SPEAK}.`,
    judge: 'a fitting reply to the message',
  },
} as const
export type Mode = keyof typeof MODES
const DASHER_NOTE = `This keyboard is different: every option is "type: X", which types the characters X (␣ is a space,
"space" types one), with the chance a typical writer would type it next. End each word with a space; ${SPEAK} is
offered once the last word is finished.`
const LETTERS_NOTE = `This keyboard offers no word predictions until you type: type letters (some keys type a common
two-letter pair), and words starting with what you've typed appear to pick from.`
export const instructionsFor = (mode: Mode = 'answer', keyboard: Keyboard = DEFAULT_KEYBOARD) =>
  `${instructions}\n${DASHER ? `${DASHER_NOTE}\n` : keyboard === 'letters' ? `${LETTERS_NOTE}\n` : ''}${MODES[mode].instruction}`

// One choice call: which kind of reply the message calls for.
// One choice call: which kind of reply the message calls for and, with chooseKeyboard, which
// keyboard Jev wants to write it with.
export async function classify(
  question: string,
  conversation?: Post[],
  chooseKeyboard = false,
): Promise<{ mode: Mode; confidence: number; keyboard?: Keyboard; keyboardConfidence?: number }> {
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question },
    questions: {
      mode: choice(
        'What kind of reply does the message call for?',
        Object.fromEntries(Object.entries(MODES).map(([k, m]) => [k, `reply this way when ${m.when}`])),
      ),
      ...(chooseKeyboard
        ? {
            keyboard: choice(
              'You will write your reply one menu pick at a time. Which keyboard would you rather write it with?',
              Object.fromEntries(Object.entries(KEYBOARDS).map(([k, d]) => [k, d])),
            ),
          }
        : {}),
    },
  })
  const a = r.answers.mode as { choice: Mode; confidence: number }
  const k = r.answers.keyboard as { choice: Keyboard; confidence: number } | undefined
  return { mode: a.choice, confidence: a.confidence, keyboard: k?.choice, keyboardConfidence: k?.confidence }
}
// Drops thread posts Jev rates irrelevant to the latest message, so stray topics (a horse joke five
// posts up) don't leak into its word choices. Always kept: the latest message, the post it replies
// to, Jev's own latest reply, linked posts, and gap markers. One call; a short thread is untouched.
export const RELEVANT = 0.5
export async function relevantContext(conversation: Post[]): Promise<{ kept: Post[]; dropped: Post[] }> {
  const last = conversation.length - 1
  const threadIdx = conversation.map((p, i) => i).filter((i) => i < last && !p_isLinked(conversation[i]))
  const parent = threadIdx.at(-1)
  const lastOwn = [...threadIdx].reverse().find((i) => conversation[i].author === 'you')
  const always = new Set([last, parent, lastOwn].filter((i): i is number => i !== undefined))
  const scored = threadIdx.filter((i) => !always.has(i) && conversation[i].author !== '…')
  if (!scored.length) return { kept: conversation, dropped: [] }
  const r = await jev().systemOne({
    state: {
      latest_message: conversation[last],
      earlier_posts: Object.fromEntries(scored.map((i) => [`p${i}`, conversation[i]])),
    },
    questions: Object.fromEntries(
      scored.map((i) => [`p${i}`, noul(`Is earlier post \`p${i}\` relevant to understanding or replying to the latest message?`)]),
    ),
  })
  const keep = (i: number) => always.has(i) || !scored.includes(i) || (r.answers[`p${i}`] as { noul: number }).noul >= RELEVANT
  return { kept: conversation.filter((_, i) => keep(i)), dropped: conversation.filter((_, i) => !keep(i)) }
}
const p_isLinked = (p: Post) => p.author.endsWith('(linked post)')

const HISTORY = 10
let client: TypeSafeClient | undefined
// The SDK's timeout covers the response arriving, not reading its body, and a stalled body
// once hung the bot for good. Every call gets a hard deadline and one retry.
const CALL_DEADLINE_MS = 45_000
function newClient(): TypeSafeClient {
  const c = new TypeSafeClient()
  const call = c.systemOne.bind(c)
  const once = (...args: Parameters<typeof call>) => {
    let timer: NodeJS.Timeout | undefined
    const deadline = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`Jev call exceeded ${CALL_DEADLINE_MS / 1000}s`)), CALL_DEADLINE_MS)
    })
    return Promise.race([call(...args), deadline]).finally(() => clearTimeout(timer))
  }
  c.systemOne = ((...args: Parameters<typeof call>) => once(...args).catch(() => once(...args))) as typeof c.systemOne
  return c
}
export const jev = () => (client ??= newClient())

// One draft in progress: its text, the letters typed toward the next word, and what it has rejected.
export type Branch = {
  words: string[]
  // How many leading words are fixed (a prefilled verdict) and can't be backspaced.
  locked: number
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

// Every punctuation mark on a US keyboard, plus en and em dashes and an ellipsis.
const PUNCTUATION = [...'.,?!:;\'"-–—…()[]{}/\\&*@#$%^+=_~<>|`']

export type MenuOptions = { words: number; common: number; minWordsToSpeak: number; phrases?: number; keyboard?: Keyboard; extra?: string[] }

export const newBranch = (words: string[] = []): Branch => ({ words: [...words], locked: words.length, prefix: '', undo: [], rejected: new Map(), steps: [], rejections: 0, score: 1 })
export const cloneBranch = (b: Branch): Branch => ({
  ...b,
  words: [...b.words],
  undo: [...b.undo],
  rejected: new Map([...b.rejected].map(([k, v]) => [k, new Set(v)])),
  steps: [...b.steps],
})
export const branchKey = (b: Branch) => `${b.words.join(' ')}|${b.prefix}`
// Words joined by spaces, except that a run of closing punctuation sticks to a preceding word that
// ends in a letter or digit ("no." not "no ."), while punctuation-only words keep their spaces
// ("... --- ...").
export const branchText = (b: Branch) =>
  b.words.reduce((text, w, i) => (i > 0 && /^[.,!?;:…)\]}%'"]+$/.test(w) && /[a-z0-9]$/i.test(b.words[i - 1]) ? text + w : text ? `${text} ${w}` : w), '')
const reject = (b: Branch, key: string, option: string) => b.rejected.set(key, (b.rejected.get(key) ?? new Set()).add(option))

export function menuFor(b: Branch, o: MenuOptions): Record<string, string> {
  if (DASHER) {
    const menu = dasherMenu(`${branchText(b)}${b.words.length ? ' ' : ''}${b.prefix}`)
    if (b.prefix || b.words.length > b.locked) menu.backspace = 'delete the last character'
    if (b.words.length && !b.prefix) menu[SPEAK] = 'finish and speak the text aloud'
    for (const opt of b.rejected.get(branchKey(b)) ?? []) delete menu[opt]
    return menu
  }
  const { words, prefix } = b
  // A phrase is stored as one entry; predictions follow its last word.
  // After a punctuation-only word (or nothing), predictions start afresh, as at a sentence start.
  const prev = words.length && /[a-z0-9]/i.test(words.at(-1)!) ? words.at(-1)!.split(' ').at(-1)!.replace(/[^a-z0-9']+$/i, '') : START
  const menu: Record<string, string> = {}
  const keyboard = o.keyboard ?? DEFAULT_KEYBOARD
  const predicted = predict(prev, prefix, o.words, o.common, keyboard)
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
  // Words from the conversation the predictor doesn't know (names, jargon), offered once they match.
  for (const w of o.extra ?? []) if (w.startsWith(prefix)) menu[`word: ${w}`] ??= `append "${w}" (from the conversation)`
  // A lone letter is only a word if it's "a", "i" or a digit; otherwise committing it leaves a
  // stray "s" where Jev was heading for a word the predictions hadn't shown yet.
  const committable = prefix.length > 1 || /^[ai0-9]$/i.test(prefix)
  if (committable && !isBlocked(prefix)) menu[`word: ${prefix}`] ??= `enter "${prefix}" as typed`
  for (const l of keyboard === 'letters' ? [...letters, ...PAIRS] : letters) menu[`letter: ${l}`] = `narrow predictions to words starting "${prefix + l}"`
  for (const p of PUNCTUATION) menu[`key: ${p}`] = `type "${p}"`
  if (committable) menu[SPACE] = `end the word "${prefix}" as typed`
  if (prefix || words.length > b.locked) menu.backspace = prefix ? `delete the typed letter "${prefix.at(-1)}"` : `delete "${words.at(-1)}"`
  if (words.filter((w) => /\w/.test(w)).length >= o.minWordsToSpeak && !prefix) menu[SPEAK] = 'finish and speak the text aloud'
  for (const opt of b.rejected.get(branchKey(b)) ?? []) delete menu[opt]
  return menu
}

// Applies a non-SPEAK pick to the branch.
export function applyPick(b: Branch, pick: string) {
  if (pick === 'backspace') {
    const last = b.undo.pop()
    if (last) reject(b, last.from, last.option)
    if (b.prefix) b.prefix = b.prefix.slice(0, -1)
    else if (DASHER) {
      // Character by character: deleting the space after a word reopens that word.
      const w = b.words.pop()
      if (w && !/^[.,?]$/.test(w)) b.prefix = w
    } else b.words.pop()
    return
  }
  if (pick.startsWith('type: ')) {
    b.undo.push({ from: branchKey(b), option: pick })
    for (const ch of chunkOf(pick)) {
      if (ch === ' ' || /[.,?]/.test(ch)) {
        if (b.prefix) b.words.push(b.prefix)
        b.prefix = ''
        if (ch !== ' ') b.words.push(ch)
      } else b.prefix += ch
    }
    return
  }
  b.undo.push({ from: branchKey(b), option: pick })
  if (pick.startsWith('letter: ')) b.prefix += pick.slice(8)
  else if (pick.startsWith('key: ')) b.prefix += pick.slice(5)
  else if (isSpace(pick)) {
    if (b.prefix) b.words.push(b.prefix)
    b.prefix = ''
  }
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
  const { keyboard = DEFAULT_KEYBOARD, mode = 'answer', phrases = 0, conversation, extraInstructions, prefill = [], extraWords = [], minWordsToSpeak = 1, confirmSpeak = false, maxRejections = 2, maxSteps = 100, words: nWords = 150, common = 30, dryRun = false, log = () => {} } = opts
  client ??= newClient()
  const b = newBranch(prefill)

  for (let step = 0; step < maxSteps; step++) {
    const menu = menuFor(b, { words: nWords, common, minWordsToSpeak: minWordsToSpeak + prefill.length, phrases, keyboard, extra: extraWords })
    const state = { instructions: extraInstructions ? `${instructionsFor(mode, keyboard)}\n${extraInstructions}` : instructionsFor(mode, keyboard), ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, recent_actions: recentActions(b), text_so_far: branchText(b) || '(nothing yet)', letters_typed: b.prefix || '(none)' }
    const questions = { next: choice('Which menu option do you pick next?', menu) }
    if (dryRun) {
      console.log(JSON.stringify({ state, questions }, null, 2))
      break
    }
    const r = await client.systemOne({ state, questions })
    const a = r.answers.next as { choice: string; confidence: number; probabilities: Record<string, number> }
    b.steps.push({ menu: Object.keys(menu), pick: a.choice, confidence: a.confidence, probabilities: a.probabilities })
    log(`${String(step + 1).padStart(2)} ${a.choice.padEnd(16)} conf=${a.confidence.toFixed(2)}  | ${branchText(b)}${b.prefix ? ' ' + b.prefix + '…' : ''}`)

    if (isSpeak(a.choice)) {
      if (!confirmSpeak || b.rejections >= maxRejections) break
      // A second opinion on stopping: a "no" withdraws SPEAK from this point and Jev carries on.
      const c = await client.systemOne({ state, questions: { final: noul('Is the text so far your final answer?') } })
      const final = (c.answers.final as { noul: number }).noul
      b.steps.at(-1)!.final = final
      log(`   final answer? ${final.toFixed(2)}`)
      if (final >= 0.5) break
      reject(b, branchKey(b), SPEAK)
      b.rejections++
      // With no checks left, SPEAK is open again everywhere; leaving old blocks in place sent
      // Jev on detours, deleting words just to reach a text it was allowed to finish.
      if (b.rejections >= maxRejections) for (const set of b.rejected.values()) set.delete(SPEAK)
      continue
    }
    applyPick(b, a.choice)
  }
  const last = b.steps.at(-1)
  return { question, answer: branchText(b), finished: !!last && isSpeak(last.pick) && (last.final ?? 1) >= 0.5, steps: b.steps }
}

export function save(t: Talk): string {
  mkdirSync('transcripts', { recursive: true })
  const out = `transcripts/${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  writeFileSync(out, JSON.stringify(t, null, 2))
  return out
}

// A bare yes or no ("no", "no.", "no. no", "not"): allowed, but it has to clear a higher bar to win.
export const isBare = (text: string) => /^\s*((yes|yep|yeah|no|nope|not|maybe)[\s.,?!]*)+$/i.test(text)

// Words in the conversation the predictor has never seen ("femshep", a name, a coinage), so Jev
// can pick them whole instead of spelling them. With judge, Jev keeps only the ones it might use.
export const MAX_CONTEXT_WORDS = 30
export function unfamiliarWords(texts: string[]): string[] {
  const seen = new Set<string>()
  for (const t of texts)
    for (const m of t.replace(/[‘’]/g, "'").replace(/https?:\S+|@\S+|\[image: [^\]]*\]/g, ' ').toLowerCase().matchAll(/[a-z][a-z0-9'-]*[a-z0-9]/g)) {
      const w = m[0]
      if (w.length >= 3 && !unigram.has(w) && !containsBlocked(w)) seen.add(w)
    }
  return [...seen].slice(0, MAX_CONTEXT_WORDS)
}
export async function usefulWords(question: string, conversation: Post[] | undefined, words: string[]): Promise<{ word: string; p: number }[]> {
  if (!words.length) return []
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question, words },
    questions: Object.fromEntries(words.map((w, i) => [`w${i}`, noul(`Might you use the word "${w}" in your reply to the message?`)])),
  })
  return words.map((word, i) => ({ word, p: (r.answers[`w${i}`] as { noul: number }).noul }))
}

// Jev's own verdict on a yes-or-no question, as the probability of yes. Asked fresh, without the
// keyboard, so it is the plainest Jev output there is; JT then types the justification.
export const UNSURE = 0.1
export async function verdict(question: string, conversation?: Post[]): Promise<{ yes: number; word: string }> {
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question },
    questions: { yes: noul('Is the answer to the yes-or-no question in the message yes?') },
  })
  const yes = (r.answers.yes as { noul: number }).noul
  return { yes, word: Math.abs(yes - 0.5) < UNSURE ? 'maybe' : yes >= 0.5 ? 'yes' : 'no' }
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
