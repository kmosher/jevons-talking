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
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { gunzipSync } from 'node:zlib'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { choice, noul, score, TypeSafeClient } from '@typesafe-ai/sdk'
import { thesaurus } from './wordnet.ts'
import { chunkOf, dasherMenu } from './dasher.ts'

export type Step = { menu: string[]; pick: string; confidence: number; probabilities: Record<string, number>; final?: number; fix?: number; cased?: string; scores?: Scores; complete?: number; stuck?: number; pruned?: number }
export type Talk = { question: string; answer: string; finished: boolean; steps: Step[] }
// A context entry. Images (descriptions plus alt text) and JT's rejected drafts ride in their
// own fields, not inside the text, so Jev can tell what was said from what was seen or tried.
export type Role = 'thread' | 'you' | 'linked' | 'asker' | 'gap'
export type Post = { author: string; role?: Role; text: string; images?: string[]; rejected_drafts?: string[] }
type Options = { keyboard?: Keyboard; mode?: Mode; phrases?: number; conversation?: Post[]; extraInstructions?: string; prefill?: string[]; extraWords?: string[]; banned?: Set<string>; minWordsToSpeak?: number; confirmSpeak?: boolean; maxRejections?: number; maxSteps?: number; words?: number; common?: number; dryRun?: boolean; log?: (line: string) => void }

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
// With JEV_KEY_STYLE=angle, letter and punctuation keys are shown to Jev as <a> and <?> instead
// of "letter: a" and "key: ?", like the <end-word> control keys. Only the names Jev sees change:
// presentMenu renames a menu on the way out, and the returned map turns Jev's answer back.
const ANGLE = process.env.JEV_KEY_STYLE === 'angle'
export const shown = (o: string) => (!ANGLE ? o : o.startsWith('letter: ') ? `<${o.slice(8)}>` : o.startsWith('key: ') ? `<${o.slice(5)}>` : o)
export function presentMenu(menu: Record<string, string>): { menu: Record<string, string>; back: (o: string) => string } {
  if (!ANGLE) return { menu, back: (o) => o }
  const names = new Map(Object.keys(menu).map((o) => [shown(o), o]))
  return { menu: Object.fromEntries(Object.entries(menu).map(([o, d]) => [shown(o), d])), back: (o) => names.get(o) ?? o }
}
// A choice answer with its option names turned back into the menu's own.
export function unpresent<T extends { choice: string; probabilities: Record<string, number> }>(a: T, back: (o: string) => string): T {
  return { ...a, choice: back(a.choice), probabilities: Object.fromEntries(Object.entries(a.probabilities).map(([o, p]) => [back(o), p])) }
}
// Spell out the cost of a near-miss (JEV_LETTER_HINT=off drops this): picking a word that isn't quite
// right and deleting it takes two picks, while one or two letters bring up better candidates.
// JEV_OWN_WORDS=on: ask for the reply in Jev's own words rather than the question's.
const OWN_WORDS =
  process.env.JEV_OWN_WORDS === 'on'
    ? `
  Say it in your own words rather than repeating the question's words back; spelling out the word
  you actually want is fine.`
    : ''
const LETTER_HINT = process.env.JEV_LETTER_HINT !== 'off'
  ? `
  If none of the predicted words is the one you want, don't pick a near miss and delete it later:
  type its first letter or two, and the predictions will refill with words that start that way.`
  : ''
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
- "${ANGLE ? '<x>' : 'letter: X'}" — type a letter to narrow the word predictions to words starting with what you've typed
  (use this when the word you want is not among the predictions). Any letter or digit can be typed, so
  you can spell any word, then enter it as typed.${LETTER_HINT}${OWN_WORDS}
- "${ANGLE ? '<.>' : 'key: X'}" — type a punctuation mark${ANGLE ? ' (the mark between the brackets)' : ' X'}: every mark on a keyboard, plus — and …, and any mark can be
  typed as often as you like.
- "${SPACE}" — end the word you're typing, exactly as typed (after letters, punctuation or both).
- "backspace" (undo the last character typed, or else the last word), and "${SPEAK}" (finish: your text is
  spoken aloud as your final answer).
recent_actions lists your last few picks. Options you already backspaced over from the current text
are not offered again: if you're stuck, backspace further and rephrase.
If conversation_so_far is present, the question is the latest message in that conversation (role
"asker"). Role "you" marks your own earlier replies, with any drafts you passed over in rejected_drafts;
role "linked" marks posts that were linked or quoted; "images" holds descriptions of a post's images.
"@you" in a message means it is addressed to you.`

// What kind of reply a message calls for, chosen by Jev before it writes anything. Each mode
// sets the last line of the instructions and the wording the judges rate drafts against.
// Yes/no answers ask for a reason with personality (JEV_YESNO_STYLE=plain: a plain justification);
// on 10 questions it gave "no, it's a ball" for the flat earth and scored higher in 7.
const VIVID = process.env.JEV_YESNO_STYLE !== 'plain'
export const MODES = {
  answer: {
    when: 'it asks a question that can be answered',
    instruction: `Aim for a short, correct answer of one or two sentences, then pick ${SPEAK}.`,
    judge: 'a good answer to the question',
  },
  comeback: {
    when: 'it challenges, teases, insults or provokes you, or makes a joke at your expense',
    instruction: `The message is not really a question: reply with a short, witty comeback, then pick ${SPEAK}.`,
    judge: 'a good, witty reply to the message',
  },
  react: {
    when: 'it shares something (a link, image, post, news, a remark or an opinion) and invites your reaction',
    instruction: `React to what was shared or said with a short, opinionated sentence, then pick ${SPEAK}.`,
    judge: 'a good reaction to what was shared or said',
  },
  scene: {
    when: 'it sets a scene, plays a game, or asks you to imagine, describe or role-play something',
    instruction: `Play along: say in character what you see, do or feel, in a short sentence, then pick ${SPEAK}.`,
    judge: 'a good in-character reply that plays along',
  },
  yesno: {
    when: 'it is a yes-or-no question',
    instruction: VIVID
      ? `Your verdict on the yes-or-no question is already typed. Back it up in a few words with some personality: a vivid comparison, a surprising detail or a joke, then pick ${SPEAK}.`
      : `Your verdict on the yes-or-no question is already typed. Justify it in a few words, then pick ${SPEAK}.`,
    judge: VIVID ? 'a good, colorful answer to the yes-or-no question, with a reason' : 'a good answer to the yes-or-no question, with a reason',
  },
  ask: {
    when: 'it asks you to ask something, or the best reply is a question of your own back',
    instruction: `Reply with a short question of your own, then pick ${SPEAK}.`,
    judge: 'a good question to ask in reply',
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

// One call before writing: which kind of reply the message calls for (and, with chooseKeyboard,
// which keyboard Jev wants), plus the verdict and context-word questions, asked up front so they
// don't cost calls of their own. The verdict is only used if the mode turns out to be yesno.
export type Prepared = {
  mode: Mode
  confidence: number
  keyboard?: Keyboard
  keyboardConfidence?: number
  verdict: { yes: number; know: number; word: string }
  contextWords: { word: string; p: number }[]
  // Whether the message asks for something different from JT's earlier replies (asked only
  // when the thread has some).
  novelty?: number
}
export async function classify(question: string, conversation?: Post[], chooseKeyboard = false, words: string[] = []): Promise<Prepared> {
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question, ...(words.length ? { words } : {}) },
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
      ...VERDICT_QUESTIONS,
      ...wordQuestions(words),
      ...(conversation?.some((p) => p.role === 'you') ? { novelty: noul('Is the message pushing back on your earlier reply, or asking you to say something new instead of repeating yourself?') } : {}),
    },
  })
  const a = r.answers.mode as { choice: Mode; confidence: number }
  const k = r.answers.keyboard as { choice: Keyboard; confidence: number } | undefined
  return {
    mode: a.choice,
    confidence: a.confidence,
    keyboard: k?.choice,
    keyboardConfidence: k?.confidence,
    verdict: readVerdict(r.answers),
    contextWords: readWords(r.answers, words),
    novelty: (r.answers as Record<string, { noul?: number }>).novelty?.noul,
  }
}
// Drops thread posts Jev rates irrelevant to the latest message, so stray topics (a horse joke five
// posts up) don't leak into its word choices. Always kept: the latest message, the post it replies
// to, Jev's own latest reply, linked posts, and gap markers. One call; a short thread is untouched.
export const RELEVANT = 0.5
export async function relevantContext(conversation: Post[]): Promise<{ kept: Post[]; dropped: Post[]; trimmed: string[] }> {
  const last = conversation.length - 1
  const threadIdx = conversation.map((_, i) => i).filter((i) => i < last && !p_isLinked(conversation[i]))
  const parent = threadIdx.at(-1)
  const lastOwn = [...threadIdx].reverse().find((i) => conversation[i].author === 'you')
  const always = new Set([last, parent, lastOwn].filter((i): i is number => i !== undefined))
  const scored = threadIdx.filter((i) => !always.has(i) && conversation[i].author !== '…')
  // Each post's extras are rated on their own: most replies don't need the drafts JT passed over.
  const withDrafts = conversation.map((_, i) => i).filter((i) => conversation[i].rejected_drafts?.length)
  if (!scored.length && !withDrafts.length) return { kept: conversation, dropped: [], trimmed: [] }
  const r = await jev().systemOne({
    state: {
      latest_message: conversation[last],
      earlier_posts: Object.fromEntries([...new Set([...scored, ...withDrafts])].sort((a, b) => a - b).map((i) => [`p${i}`, conversation[i]])),
    },
    questions: Object.fromEntries([
      ...scored.map((i) => [`p${i}`, noul(`Is earlier post \`p${i}\` relevant to understanding or replying to the latest message?`)]),
      ...withDrafts.map((i) => [`d${i}`, noul(`Would the rejected_drafts of post \`p${i}\` (drafts you passed over) help you reply to the latest message?`)]),
    ]),
  })
  const v = (k: string) => (r.answers[k] as { noul: number }).noul
  const keep = (i: number) => always.has(i) || !scored.includes(i) || v(`p${i}`) >= RELEVANT
  const trimmed: string[] = []
  const kept = conversation
    .map((p, i) => {
      if (!withDrafts.includes(i) || v(`d${i}`) >= RELEVANT) return p
      trimmed.push(`drafts of "${p.text.slice(0, 30)}"`)
      const { rejected_drafts: _, ...rest } = p
      return rest
    })
    .filter((_, i) => keep(i))
  return { kept, dropped: conversation.filter((_, i) => !keep(i)), trimmed }
}
const p_isLinked = (p: Post) => p.role === 'linked' || p.author.endsWith('(linked post)')

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
  const retried = (...args: Parameters<typeof call>) => once(...args).catch(() => once(...args))
  c.systemOne = (CACHE_DIR ? cached(retried) : retried) as typeof c.systemOne
  return c
}
export const jev = () => (client ??= newClient())
// Record and replay (JEV_CACHE=<dir>, for experiments): each answer is saved under a hash of the
// exact request, so a repeat run, or a variant that only changes a late stage, pays only for the
// requests that differ. Two variants also see the same answers up to where they diverge.
const CACHE_DIR = process.env.JEV_CACHE
export const cacheStats = { hits: 0, misses: 0 }
function cached<A extends unknown[], R>(call: (...args: A) => Promise<R>): (...args: A) => Promise<R> {
  mkdirSync(CACHE_DIR!, { recursive: true })
  return async (...args) => {
    const file = join(CACHE_DIR!, `${createHash('sha256').update(JSON.stringify(args)).digest('hex')}.json`)
    if (existsSync(file)) {
      cacheStats.hits++
      return JSON.parse(readFileSync(file, 'utf8')) as R
    }
    cacheStats.misses++
    const r = await call(...args)
    writeFileSync(file, JSON.stringify(r))
    return r
  }
}

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
  // A word Jev said it would replace if it could: the next menu offers swaps for it.
  // How many times a short draft's SPEAK was checked for completeness (JEV_SPEAK_GATE).
  gateChecks?: number
  // Jev said it was stuck and the draft ended there (JEV_STUCK).
  stopped?: boolean
  swap?: { idx: number; word: string; kinds: SwapKind[] }
  // Words Jev said it would replace, with how strongly (JEV_BEAM_FLAG ranks branches by them).
  flaws?: { idx: number; word: string; p: number }[]
  // The word whose casing Jev was last asked about (JEV_WORD_CASE).
  caseAsked?: number
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
const EMOJI_ON = process.env.JEV_EMOJI === 'on'
const EMOJI = [...'🙂😂🤔😢😡😱👍👎🙏🎉🔥✨👀🤖🐢🌙💀🤷'].filter((c) => c.trim())
const PUNCTUATION = [...'.,?!:;\'"-–—…()[]{}/\\&*@#$%^+=_~<>|`']

export type MenuOptions = { words: number; common: number; minWordsToSpeak: number; phrases?: number; keyboard?: Keyboard; extra?: string[]; banned?: Set<string> }

export const newBranch = (words: string[] = []): Branch => ({ words: [...words], locked: words.length, prefix: '', undo: [], rejected: new Map(), steps: [], rejections: 0, score: 1 })
export const cloneBranch = (b: Branch): Branch => ({
  ...b,
  words: [...b.words],
  undo: [...b.undo],
  rejected: new Map([...b.rejected].map(([k, v]) => [k, new Set(v)])),
  steps: [...b.steps],
  flaws: b.flaws && [...b.flaws],
})
export const branchKey = (b: Branch) => `${b.words.join(' ')}|${b.prefix}`
// Words joined by spaces, except that a run of closing punctuation sticks to a preceding word that
// ends in a letter or digit ("no." not "no ."), while punctuation-only words keep their spaces
// ("... --- ...").
export const branchText = (b: Branch) =>
  b.words.reduce((text, w, i) => (i > 0 && /^[.,!?;:…)\]}%'"]+$/.test(w) && /[a-z0-9]$/i.test(b.words[i - 1]) ? text + w : text ? `${text} ${w}` : w), '')
const joinWords = (words: string[]) => branchText({ words } as Branch)
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
  const prev = words.length && /[a-z0-9]/i.test(words.at(-1)!) ? words.at(-1)!.split(' ').at(-1)!.replace(/[^a-z0-9']+$/i, '').toLowerCase() : START
  const menu: Record<string, string> = {}
  const keyboard = o.keyboard ?? DEFAULT_KEYBOARD
  const predicted = predict(prev, prefix, o.words, o.common, keyboard)
  // Each word shows the text it would make, so Jev judges the result before
  // picking rather than after (most deletes undid the word picked the step before). JEV_PREVIEW=off restores "append X".
  const before = PREVIEW ? branchText(b) : ''
  for (const w of predicted) menu[`word: ${w}`] = PREVIEW ? `→ "${before ? `${before} ` : ''}${w}"` : `append "${w}"`
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
  // Replacements for a word Jev just flagged, from the thesaurus (rewrite-as-you-type).
  if (b.swap && !prefix && SWAP_ASK === 'off') for (const w of swapCandidates(b.swap)) menu[`swap: ${w}`] ??= `replace "${b.swap.word}" with "${w}"`
  // Words from the conversation the predictor doesn't know (names, jargon), offered once they match.
  for (const w of o.extra ?? []) if (w.startsWith(prefix)) menu[`word: ${w}`] ??= `append "${w}" (from the conversation)`
  // A lone letter is only a word if it's "a", "i" or a digit; otherwise committing it leaves a
  // stray "s" where Jev was heading for a word the predictions hadn't shown yet.
  const committable = prefix.length > 1 || /^[ai0-9]$/i.test(prefix)
  if (committable && !isBlocked(prefix)) menu[`word: ${prefix}`] ??= `enter "${prefix}" as typed`
  for (const l of keyboard === 'letters' ? [...letters, ...PAIRS] : letters) menu[`letter: ${l}`] = `narrow predictions to words starting "${prefix + l}"`
  for (const p of PUNCTUATION) menu[`key: ${p}`] = `type "${p}"`
  // With JEV_EMOJI=on, a few emoji, each added as a word of its own.
  if (EMOJI_ON) for (const e of EMOJI) menu[`word: ${e}`] ??= `add the emoji ${e}`
  if (committable) menu[SPACE] = `end the word "${prefix}" as typed`
  if (prefix || words.length > b.locked) menu.backspace = prefix ? `delete the typed letter "${prefix.at(-1)}"` : `delete "${words.at(-1)}"`
  if (words.filter((w) => /\w/.test(w)).length >= o.minWordsToSpeak && !prefix) menu[SPEAK] = 'finish and speak the text aloud'
  for (const opt of b.rejected.get(branchKey(b)) ?? []) delete menu[opt]
  // Word pairs JT already used in the thread, when the message asks for something new (see repeatedPairs).
  if (o.banned?.size && !prefix) {
    const prev2 = words.length > 1 ? words.at(-2)!.toLowerCase() : ''
    for (const k of Object.keys(menu)) {
      const w = k.startsWith('word: ') ? k.slice(6).split(' ')[0].toLowerCase() : ''
      if (w && (o.banned.has(`${prev} ${w}`) || o.banned.has(`${prev2} ${prev} ${w}`))) delete menu[k]
    }
  }
  // Jev takes at most 255 options; the lowest-ranked predictions make room for swaps and thread words.
  const over = Object.keys(menu).length - MAX_OPTIONS
  if (over > 0) for (const w of predicted.slice(-over).reverse()) delete menu[`word: ${w}`]
  return menu
}
const MAX_OPTIONS = 255

// Applies a non-SPEAK pick to the branch.
export function applyPick(b: Branch, pick: string) {
  const swap = b.swap
  b.swap = undefined
  if (pick.startsWith('swap: ') && swap) {
    b.undo.push({ from: branchKey(b), option: pick })
    b.words[swap.idx] = pick.slice(6)
    return
  }
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

// Rewrite-as-you-type: each pick call also asks whether Jev would replace the last word; a "yes"
// puts thesaurus swaps for it on the next menu. Off by default (JEV_SWAP=on): Jev flagged words
// but almost never took a swap, and the final rewrite makes the edits that stick.
export const SWAP = process.env.JEV_SWAP === 'on'
export const SWAPS = 20
export const SWAP_BAR = 0.6
// Other forms of a word: tense, person and number for the common irregulars, then the regular
// -s/-ed/-ing endings. Swaps offer these alongside synonyms, so "i is" can become "i am".
const FORM_SETS = [
  'be am is are was were been being', 'have has had having', 'do does did done doing', 'go goes went gone going',
  'i me my mine myself', 'we us our ours', 'he him his', 'she her hers', 'they them their theirs', 'it its',
  'can could', 'will would', 'shall should', 'may might', 'this these', 'that those', 'a an the',
  "isn't aren't wasn't weren't", "don't doesn't didn't", "can't couldn't", "won't wouldn't",
].map((g) => g.split(' '))
export function forms(word: string): string[] {
  const w = word.toLowerCase()
  const set = FORM_SETS.find((g) => g.includes(w))
  if (set) return set.filter((x) => x !== w)
  const stem = w.replace(/(ing|ed|es|s)$/, '')
  return [stem, `${stem}s`, `${stem}ed`, `${stem}ing`].filter((x) => x !== w && unigram.has(x))
}
// Function words ("i", "is", "can") get flagged at about the base rate; they're only swappable
// when they have other forms, and then only those are offered.
const FUNCTION_WORDS = new Set('i me my you your he she it its we they them a an the is am are was were be been being do does did have has had can could will would shall should may might must not no and or but if so to of in on at by for with from as that this these those there here what which who how why when where'.split(' '))
// The last word, if it's one that could be swapped: a plain word past any prefilled verdict.
export function swappable(b: Branch): { idx: number; word: string } | undefined {
  if (!SWAP || b.swap) return undefined
  return lastWord(b)
}
// The last word, if it's one Jev could be asked about.
export function lastWord(b: Branch): { idx: number; word: string } | undefined {
  if (b.prefix) return undefined
  const idx = b.words.length - 1
  const word = b.words[idx]
  return idx >= b.locked && word && /^[a-z][a-z'-]*$/i.test(word) && (!FUNCTION_WORDS.has(word.toLowerCase()) || forms(word).length > 0) && b.steps.at(-1)?.pick.startsWith('swap: ') !== true ? { idx, word } : undefined
}
// With JEV_BEAM_FLAG=on the beam asks the replace-the-last-word question of every branch and
// ranks a branch down by its worst flagged word still in the text, instead of offering swaps.
export const BEAM_FLAG = process.env.JEV_BEAM_FLAG === 'on'
const FLAW_FLOOR = 0.35
const FLAW_WEIGHT = 0.5
export const flawFactor = (b: Branch) => {
  const worst = Math.max(0, ...(b.flaws ?? []).filter((f) => b.words[f.idx] === f.word).map((f) => f.p))
  return 1 - FLAW_WEIGHT * Math.max(0, (worst - FLAW_FLOOR) / (1 - FLAW_FLOOR))
}
// Each pick call also asks (JEV_WORD_CASE=off to stop) how the last word should be written (as is,
// Capitalized or ALL CAPS) and applies the answer at once, in place of one choice for the
// whole reply at the end.
export const WORD_CASE = process.env.JEV_WORD_CASE !== 'off'
const WORD_CASES = { as_is: (w: string) => w, capitalized: (w: string) => w[0].toUpperCase() + w.slice(1), all_caps: (w: string) => w.toUpperCase() }
export function caseTarget(b: Branch): { idx: number; word: string } | undefined {
  if (!WORD_CASE || b.prefix) return undefined
  const idx = b.words.length - 1
  const word = b.words[idx]
  return idx >= b.locked && word && /^[a-z][a-z'-]*$/i.test(word) && b.caseAsked !== idx ? { idx, word } : undefined
}
export const caseQuestion = (word: string, about = 'the last word of text_so_far') =>
  choice(`How should ${about} be written?`, Object.fromEntries(Object.entries(WORD_CASES).map(([k, f]) => [k, `"${f(word)}"`])))
export function applyCase(b: Branch, target: { idx: number; word: string } | undefined, answer: { choice: string } | undefined): string | undefined {
  if (!target || !answer) return undefined
  if (b.words[target.idx] !== target.word) return undefined
  b.caseAsked = target.idx
  const f = WORD_CASES[answer.choice as keyof typeof WORD_CASES]
  if (!f || answer.choice === 'as_is') return undefined
  b.words[target.idx] = f(target.word)
  return answer.choice
}

// One question ("replace the last word?") or, with JEV_SWAP_SPLIT=on, one per kind of edit: its
// form (tense, person, number) and a synonym. Keys are suffixes on the caller's prefix.
export type SwapKind = 'form' | 'syn'
const SPLIT = process.env.JEV_SWAP_SPLIT === 'on'
export function fixQuestions(word: string, about = 'the last word of text_so_far'): Record<string, ReturnType<typeof noul>> {
  if (!SPLIT) return { fix: noul(`Would you replace ${about} ("${word}") with a better word, if you could?`) }
  return {
    fix_form: noul(`Would you change the form of ${about} ("${word}"), its tense, person or number, if you could?`),
    fix_syn: noul(`Would you swap ${about} ("${word}") for a different word with a similar meaning, if you could?`),
  }
}
// A flagged word's replacements: its other forms, then (for content words) thesaurus entries.
export function swapCandidates(swap: { word: string; kinds: SwapKind[] }): string[] {
  const fn = FUNCTION_WORDS.has(swap.word.toLowerCase())
  return [...(swap.kinds.includes('form') ? forms(swap.word) : []), ...(fn || !swap.kinds.includes('syn') ? [] : thesaurus(swap.word, SWAPS))].slice(0, SWAPS)
}
// How a flagged word's swaps are offered: on the next keyboard menu ('off'), or right away in
// a call of their own, with a keep option ('keep') or without ('force').
export const SWAP_ASK = (process.env.JEV_SWAP_ASK ?? 'off') as 'off' | 'keep' | 'force'
const KEEP_WORD = '(keep it)'
// One call for every branch with a flagged word: Jev picks its replacement (or keeps it), and the
// pick is recorded as a step. Returns whether a call was made.
export async function resolveSwaps(branches: Branch[], ctx: { question: string; conversation?: Post[] }): Promise<boolean> {
  const flagged = branches.filter((b) => b.swap && swapCandidates(b.swap).length)
  if (SWAP_ASK === 'off' || !flagged.length) return false
  const ids = flagged.map((_, i) => `d${i}`)
  const menus = flagged.map((b) => ({
    ...(SWAP_ASK === 'keep' ? { [KEEP_WORD]: `keep "${b.swap!.word}"` } : {}),
    ...Object.fromEntries(swapCandidates(b.swap!).map((w) => [`swap: ${w}`, `→ "${b.words.map((x, i) => (i === b.swap!.idx ? w : x)).join(' ')}"`])),
  }))
  const r = await jev().systemOne({
    state: {
      you: 'Jev, called JT on Bluesky, writing a reply one menu pick at a time',
      ...(ctx.conversation?.length ? { conversation_so_far: ctx.conversation } : {}),
      question: ctx.question,
      drafts: Object.fromEntries(flagged.map((b, i) => [ids[i], b.words.map((x, j) => (j === b.swap!.idx ? `[${x}]` : x)).join(' ')])),
    },
    questions: Object.fromEntries(flagged.map((b, i) => [ids[i], choice(`In draft \`${ids[i]}\`, which word do you want in place of the bracketed "${b.swap!.word}"?`, menus[i])])),
  })
  flagged.forEach((b, i) => {
    const a = r.answers[ids[i]] as { choice: string; confidence: number; probabilities: Record<string, number> }
    if (a.choice !== KEEP_WORD) {
      b.steps.push({ menu: Object.keys(menus[i]), pick: a.choice, confidence: a.confidence, probabilities: a.probabilities })
      applyPick(b, a.choice)
    } else b.swap = undefined
  })
  return true
}

// After a pick, flag the word for swapping if Jev wanted it changed and it is still there.
export function flagSwap(b: Branch, target: { idx: number; word: string } | undefined, p: { fix?: number; fix_form?: number; fix_syn?: number }) {
  if (!SWAP || !target || b.words[target.idx] !== target.word) return
  const kinds: SwapKind[] =
    p.fix !== undefined ? (p.fix >= SWAP_BAR ? ['form', 'syn'] : []) : [...((p.fix_form ?? 0) >= SWAP_BAR ? ['form' as const] : []), ...((p.fix_syn ?? 0) >= SWAP_BAR ? ['syn' as const] : [])]
  if (kinds.length) b.swap = { ...target, kinds }
}

// What Jev sees of a draft each step. With JEV_STATE=v2: the typed letters inline at a cursor,
// how many picks it has used, and the words it already tried and deleted at this point (they're
// off the menu), aimed at the pick-a-word-then-delete-it churn.
const STATE_V2 = process.env.JEV_STATE === 'v2'
const PREVIEW = process.env.JEV_PREVIEW !== 'off'
const LATE_BUDGET = process.env.JEV_PICK_BUDGET === 'late'
const PICKS_WARNING = 10
// A predicted word taken from the question has its probability multiplied by ECHO_WEIGHT before
// the pick is made, so an echo wins only when Jev clearly prefers it over everything else (1 turns
// this off). Common words ("is", "the", "you") are exempt. At 0.3, replies echoed less, rated a
// little higher and took fewer calls, since fewer drafts ended in the beam.
export const ECHO_WEIGHT = Number(process.env.JEV_ECHO_WEIGHT ?? 0.3)
const COMMON_WORDS = new Set([...unigram].sort((x, y) => y[1] - x[1]).slice(0, 150).map(([w]) => w))
const questionWords = (question: string) => new Set((question.toLowerCase().match(/[a-z][a-z0-9']*/g) ?? []).filter((w) => !COMMON_WORDS.has(w) && !['i', 'you', 'your', "you're"].includes(w)))
// When the message asks for something new ("be specific", "you already said that"), each word
// pair from JT's own earlier replies in the thread is taken off the menu, so it can't type the
// same phrase again. Pairs ending in a common word ("is a", "of the") stay, or grammar suffers.
export function repeatedPairs(conversation: Post[] | undefined): Set<string> {
  const pairs = new Set<string>()
  for (const p of conversation ?? []) {
    if (p.role !== 'you') continue
    const ws = p.text.toLowerCase().match(/[a-z0-9']+/g) ?? []
    for (let i = 1; i < ws.length; i++) {
      if (!COMMON_WORDS.has(ws[i])) pairs.add(`${ws[i - 1]} ${ws[i]}`)
      // Three-word runs are banned whatever the words, which catches "than are now".
      if (i >= 2) pairs.add(`${ws[i - 2]} ${ws[i - 1]} ${ws[i]}`)
    }
  }
  return pairs
}
export function steerFromEcho<T extends { choice: string; confidence: number; probabilities: Record<string, number> }>(a: T, question: string): T {
  if (ECHO_WEIGHT === 1) return a
  const echo = questionWords(question)
  const weigh = ([o, p]: [string, number]) => (o.startsWith('word: ') && o.slice(6).split(' ').some((w) => echo.has(w.toLowerCase())) ? p * ECHO_WEIGHT : p)
  const best = Object.entries(a.probabilities).reduce((x, y) => (weigh(y) > weigh(x) ? y : x))
  return best[0] === a.choice ? a : { ...a, choice: best[0], confidence: best[1], steered: a.choice }
}
export function typingState(b: Branch, maxSteps: number): Record<string, string | string[]> {
  const text = branchText(b)
  if (!STATE_V2) return { text_so_far: text || '(nothing yet)', letters_typed: b.prefix || '(none)' }
  const tried = [...(b.rejected.get(branchKey(b)) ?? [])].filter((o) => o.startsWith('word: ')).map((o) => o.slice(6))
  return {
    text_so_far: `${text}${text && b.prefix ? ' ' : ''}${b.prefix}▮`,
    // JEV_PICK_BUDGET=late: the pick count is only shown in the last PICKS_WARNING picks.
    ...(!LATE_BUDGET
      ? { picks_used: `${b.steps.length} of at most ${maxSteps}; most good replies take 5 to 20` }
      : maxSteps - b.steps.length <= PICKS_WARNING
        ? { picks_left: `${maxSteps - b.steps.length}: finish your reply soon` }
        : {}),
    ...(tried.length ? { already_tried_and_deleted_here: tried } : {}),
  }
}

// SPEAK on a draft of SHORT_REPLY words or fewer is checked for completeness first
// (JEV_SPEAK_GATE=off skips it); longer drafts may still stop on a fragment when struggling.
export const SPEAK_GATE = process.env.JEV_SPEAK_GATE !== 'off'
const SHORT_REPLY = 4
const MAX_GATE_CHECKS = 3
// JEV_STUCK=on asks, in each pick call from STUCK_AFTER picks on, whether Jev is stuck going in
// circles. A yes ends the draft where it stands (pruned, with JEV_PRUNE=on), and lets a short
// fragment through the SPEAK gate.
export const STUCK_CHECK = process.env.JEV_STUCK === 'on'
export const STUCK_AFTER = 8
// JEV_PRUNE=on: a draft that ends without SPEAK (stuck, or out of picks) is offered back with its
// last 0 to PRUNE_MAX words cut, and Jev picks which to send, so a run that trails off into word
// salad can fall back to where it still made sense. One call.
export const PRUNE = process.env.JEV_PRUNE === 'on'
export const PRUNE_MAX = 5
async function prune(b: Branch, question: string, conversation: Post[] | undefined, log: (l: string) => void): Promise<number> {
  const cuttable = b.words.length - Math.max(b.locked, 1)
  const versions = Array.from({ length: Math.min(PRUNE_MAX, cuttable) + 1 }, (_, k) => b.words.slice(0, b.words.length - k))
  if (versions.length < 2) return 0
  const options = Object.fromEntries(versions.map((w, k) => [`cut_${k}`, `"${joinWords(w)}"`]))
  const r = await jev().systemOne({
    state: { ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, note: 'You ran out of picks before finishing your reply. You can send it as it stands, or cut words off the end.' },
    questions: { send: choice('Which version of your reply do you send?', options) },
  })
  const k = Number((r.answers.send as { choice: string }).choice.slice(4))
  if (k) {
    log(`   pruned ${k} word(s): "${joinWords(b.words)}" -> "${joinWords(b.words.slice(0, -k))}"`)
    b.words = b.words.slice(0, -k)
    b.prefix = ''
  }
  return k
}

export const recentActions = (b: Branch) => {
  const recent = b.steps.slice(-HISTORY).map((s) => shown(s.pick))
  return recent.length ? recent : '(none)'
}

export async function talk(question: string, opts: Options = {}): Promise<Talk> {
  const { keyboard = DEFAULT_KEYBOARD, mode = 'answer', phrases = 0, conversation, extraInstructions, prefill = [], extraWords = [], banned, minWordsToSpeak = 1, confirmSpeak = false, maxRejections = 2, maxSteps = 100, words: nWords = 150, common = 30, dryRun = false, log = () => {} } = opts
  client ??= newClient()
  const b = newBranch(prefill)

  for (let step = 0; step < maxSteps; step++) {
    const menu = menuFor(b, { words: nWords, common, minWordsToSpeak: minWordsToSpeak + prefill.length, phrases, keyboard, extra: extraWords, banned })
    const state = { instructions: extraInstructions ? `${instructionsFor(mode, keyboard)}\n${extraInstructions}` : instructionsFor(mode, keyboard), ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, recent_actions: recentActions(b), ...typingState(b, maxSteps) }
    const target = swappable(b)
    const caseT = caseTarget(b)
    const shownMenu = presentMenu(menu)
    const questions = { next: choice('Which menu option do you pick next?', shownMenu.menu), ...(target ? fixQuestions(target.word) : {}), ...(caseT ? { case: caseQuestion(caseT.word) } : {}), ...(MODERATE_EVERY && b.words.length ? rubricQuestions('text_so_far, as your reply so far') : {}), ...(STUCK_CHECK && b.steps.length >= STUCK_AFTER ? { stuck: noul('Looking at recent_actions and text_so_far, are you stuck going in circles, so it would be better to stop and send what you have?') } : {}) }
    if (dryRun) {
      console.log(JSON.stringify({ state, questions }, null, 2))
      break
    }
    const r = await client.systemOne({ state, questions })
    const a = steerFromEcho(unpresent(r.answers.next as { choice: string; confidence: number; probabilities: Record<string, number> }, shownMenu.back), question)
    if ('steered' in a) log(`   echo: "${String(a.steered).slice(6)}" from the question; took ${a.choice} instead`)
    const fixes = Object.fromEntries(['fix', 'fix_form', 'fix_syn'].filter((k) => k in r.answers).map((k) => [k, ((r.answers as Record<string, unknown>)[k] as { noul: number }).noul])) as { fix?: number; fix_form?: number; fix_syn?: number }
    const fixVals = Object.values(fixes)
    const fix = fixVals.length ? Math.max(...fixVals) : undefined
    const stepScores = MODERATE_EVERY ? readScores(r.answers as Record<string, unknown>) : undefined
    b.steps.push({ menu: Object.keys(menu), pick: a.choice, confidence: a.confidence, probabilities: a.probabilities, ...(fix !== undefined ? { fix } : {}), ...(stepScores ? { scores: stepScores } : {}), ...('stuck' in r.answers ? { stuck: (r.answers.stuck as { noul: number }).noul } : {}) })
    log(`${String(step + 1).padStart(2)} ${a.choice.padEnd(16)} conf=${a.confidence.toFixed(2)}  | ${branchText(b)}${b.prefix ? ` ${b.prefix}…` : ''}`)

    // The last word's casing is answered in the same call as SPEAK, so apply it before stopping.
    if (isSpeak(a.choice)) {
      const casedLast = applyCase(b, caseT, (r.answers as Record<string, unknown>).case as { choice: string } | undefined)
      if (casedLast) {
        b.steps.at(-1)!.cased = casedLast
        log(`   cased "${caseT!.word}" ${casedLast}`)
      }
      // A short draft may only stop if Jev says it's a complete thought; otherwise the pick becomes
      // the best other option from the same call (no re-ask), and SPEAK is barred at this text.
      const shortReply = b.words.filter((w) => /\w/.test(w)).length <= SHORT_REPLY
      const stuck = (b.steps.at(-1)!.stuck ?? 0) >= 0.5
      if (SPEAK_GATE && shortReply && !stuck && (b.gateChecks ?? 0) < MAX_GATE_CHECKS) {
        b.gateChecks = (b.gateChecks ?? 0) + 1
        const g = await client.systemOne({
          state: { ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, reply: branchText(b) },
          questions: { complete: noul('Is the reply a complete thought, rather than cut off mid-sentence?') },
        })
        const complete = (g.answers.complete as { noul: number }).noul
        const step = b.steps.at(-1)!
        step.complete = complete
        const next = Object.entries(a.probabilities).filter(([o]) => !isSpeak(o)).sort((x, y) => y[1] - x[1])[0]
        if (complete < 0.5 && next) {
          log(`   "${branchText(b)}" isn't complete (${(complete * 100).toFixed(0)}%); taking ${next[0]} instead`)
          reject(b, branchKey(b), SPEAK)
          step.pick = next[0]
          step.confidence = next[1]
          applyPick(b, next[0])
          continue
        }
      }
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
    if ((b.steps.at(-1)!.stuck ?? 0) >= 0.5 && b.words.length > b.locked) {
      log(`   stuck (${((b.steps.at(-1)!.stuck ?? 0) * 100).toFixed(0)}%); stopping here`)
      b.stopped = true
      break
    }
    applyPick(b, a.choice)
    const cased = applyCase(b, caseT, (r.answers as Record<string, unknown>).case as { choice: string } | undefined)
    if (cased) {
      b.steps.at(-1)!.cased = cased
      log(`   cased "${caseT!.word}" ${cased}`)
    }
    flagSwap(b, target, fixes)
    if (b.swap) log(`   would change "${b.swap.word}" (${b.swap.kinds.join('+')}; ${Object.entries(fixes).map(([k, v]) => `${k} ${((v ?? 0) * 100).toFixed(0)}%`).join(', ')})`)
    if (await resolveSwaps([b], { question, conversation })) log(`   swap: ${b.steps.at(-1)!.pick.startsWith('swap: ') ? b.steps.at(-1)!.pick : 'kept'}`)
  }
  const last = b.steps.at(-1)
  const spoke = !!last && isSpeak(last.pick) && (last.final ?? 1) >= 0.5
  if (PRUNE && !spoke && last && !dryRun) {
    const k = await prune(b, question, conversation, log)
    last.pruned = k
  }
  // A stuck stop counts as finished: Jev chose to end there.
  return { question, answer: branchText(b), finished: spoke || !!b.stopped, steps: b.steps }
}

export function save(t: Talk): string {
  mkdirSync('transcripts', { recursive: true })
  const out = `transcripts/${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  writeFileSync(out, JSON.stringify(t, null, 2))
  return out
}

// A bare yes or no ("no", "no.", "no. no", "not"): allowed, but it has to clear a higher bar to win.
export const isBare = (text: string) => /^\s*((yes|yep|yeah|no|nope|not|maybe|idk)[\s.,?!]*)+$/i.test(text)

// Slashdot-style self-moderation: a fresh Jev scores the finished reply 0-5 on each rubric.
// JEV_MODERATE=off skips it; JEV_MODERATE=every also scores the text so far on every pick.
export const RUBRICS = ['funny', 'insightful', 'informative', 'interesting'] as const
export type Scores = Record<(typeof RUBRICS)[number], number>
// Jev's score questions take a 0-5 rubric and return the expected score.
const LEVELS = (r: string) => [`not ${r} at all`, `barely ${r}`, `a little ${r}`, `fairly ${r}`, `very ${r}`, `as ${r} as a reply gets`] as const
export const rubricQuestions = (what: string, key = 'score') => Object.fromEntries(RUBRICS.map((r) => [`${key}_${r}`, score(`How ${r} is ${what}?`, LEVELS(r))]))
// Asked plainly, Jev scored nearly everything under 1. Judging as the reply's proud author, by what
// the keyboard allows, spreads the scores (best rubric averaged 1.0 -> 1.5 on 14 live replies).
export const MODERATION_NOTE =
  'You wrote this reply yourself, picking one word at a time from a tiny predictive keyboard that cannot write freely. Judge it kindly, as its proud author, by what you managed with that keyboard.'
export const readScores = (answers: Record<string, unknown>, key = 'score'): Scores | undefined => {
  if (!RUBRICS.every((r) => answers[`${key}_${r}`])) return undefined
  return Object.fromEntries(RUBRICS.map((r) => [r, (answers[`${key}_${r}`] as { score: number }).score])) as Scores
}
export async function moderate(question: string, answer: string, conversation?: Post[]): Promise<Scores> {
  const r = await jev().systemOne({
    state: { note: MODERATION_NOTE, ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, reply: answer },
    questions: rubricQuestions('your reply'),
  })
  return readScores(r.answers as Record<string, unknown>)!
}
export const MODERATE_EVERY = process.env.JEV_MODERATE === 'every'

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
const wordQuestions = (words: string[]) => Object.fromEntries(words.map((w, i) => [`w${i}`, noul(`Might you use the word "${w}" in your reply to the message?`)]))
const readWords = (answers: Record<string, unknown>, words: string[]) => words.map((word, i) => ({ word, p: (answers[`w${i}`] as { noul: number }).noul }))
export async function usefulWords(question: string, conversation: Post[] | undefined, words: string[]): Promise<{ word: string; p: number }[]> {
  if (!words.length) return []
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question, words },
    questions: wordQuestions(words),
  })
  return readWords(r.answers, words)
}

// Jev's own verdict on a yes-or-no question, as the probability of yes. Asked fresh, without the
// keyboard, so it is the plainest Jev output there is; JT then types the justification.
export const UNSURE = 0.1
// "maybe" is for a real toss-up; "idk" is for a question Jev says it doesn't know the answer to,
// which a probability near 50% alone can't tell apart.
// Asked whether the question is the kind it would know, unknowables (rain on a date in 2031, life
// on other planets) score 10-20% and questions Jev has a view on, even hedged ones like "are you
// like a horse?", 40% and up. "Do you actually know…?" pulled the hedged ones down among the unknowables.
export const IDK = 0.3
const VERDICT_QUESTIONS = {
  yes: noul('Is the answer to the yes-or-no question in the message yes?'),
  know: noul('Is the yes-or-no question in the message the kind of question you would know the answer to?'),
}
function readVerdict(answers: Record<string, unknown>): { yes: number; know: number; word: string } {
  const yes = (answers.yes as { noul: number }).noul
  const know = (answers.know as { noul: number }).noul
  return { yes, know, word: know < IDK ? 'idk' : Math.abs(yes - 0.5) < UNSURE ? 'maybe' : yes >= 0.5 ? 'yes' : 'no' }
}
export async function verdict(question: string, conversation?: Post[]): Promise<{ yes: number; know: number; word: string }> {
  const r = await jev().systemOne({
    state: { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question },
    questions: VERDICT_QUESTIONS,
  })
  return readVerdict(r.answers)
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
