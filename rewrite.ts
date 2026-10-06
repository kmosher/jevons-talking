// An optional last pass: Jev rates each word of its finished reply for "would you replace this if
// you could?", then picks a replacement for the most-wanted one from the keyboard's own
// predictions there and WordNet's synonyms and broader terms for it. A fresh judge keeps whichever version rates better. JEV_REWRITE=off disables it.
import { choice, noul } from '@typesafe-ai/sdk'
import { rateDrafts } from './beam.ts'
import { thesaurus } from './wordnet.ts'
import { fillers, forms, jev, type Mode, type Post, predict } from './talk.ts'

export const REPLACE = 0.5
const MAX_REWRITES = 2
const CANDIDATES = 40

export type Rewrite = { wants: { word: string; p: number }[]; edits: { from: string; to: string; confidence: number }[]; before: string; after: string; ratings: [number, number]; kept: 'before' | 'after'; calls: number }

const KEEP = '(keep it)'
const DROP = '(delete it)'

export async function rewrite(question: string, answer: string, opts: { conversation?: Post[]; mode?: Mode; extraWords?: string[]; locked?: number; log?: (l: string) => void } = {}): Promise<Rewrite | undefined> {
  const { conversation, mode = 'answer', extraWords = [], locked = 0, log = () => {} } = opts
  const words = answer.split(/\s+/).filter(Boolean)
  const editable = words.map((_, i) => i).filter((i) => i >= locked && /[a-z0-9]/i.test(words[i]))
  if (!editable.length) return undefined
  const state = { you: 'Jev, called JT on Bluesky', ...(conversation?.length ? { conversation_so_far: conversation } : {}), message: question, your_reply: answer, words_of_your_reply: words }
  const r = await jev().systemOne({
    state,
    questions: Object.fromEntries(editable.map((i) => [`w${i}`, noul(`Would you replace word ${i + 1} of your reply ("${words[i]}") with a better word, if you could?`)])),
  })
  const wants = editable.map((i) => ({ i, word: words[i], p: (r.answers[`w${i}`] as { noul: number }).noul }))
  log(`rewrite wants: ${wants.map((w) => `${w.word} ${(w.p * 100).toFixed(0)}%`).join(', ')}`)
  let calls = 1
  const targets = wants.filter((w) => w.p >= REPLACE).sort((a, b) => b.p - a.p).slice(0, MAX_REWRITES)
  const next = [...words]
  const edits: Rewrite['edits'] = []
  for (const t of targets) {
    const prev = t.i > 0 ? next[t.i - 1].toLowerCase().replace(/[^a-z0-9']+$/, '') : '<s>'
    const nextWord = next[t.i + 1]?.toLowerCase().replace(/[^a-z0-9']+$/, '')
    const pool = [...new Set([...forms(t.word), ...thesaurus(t.word, CANDIDATES), ...fillers(prev, nextWord, CANDIDATES), ...predict(prev, '', 20, 10), ...extraWords])].filter((w) => w !== t.word.toLowerCase())
    const options = Object.fromEntries([...(process.env.JEV_REWRITE_KEEP === "on" ? [[KEEP, `keep "${t.word}"`]] : []), [DROP, `delete "${t.word}"`], ...pool.map((w) => [`word: ${w}`, `replace "${t.word}" with "${w}"`])])
    const marked = next.map((w, i) => (i === t.i ? `[${w}]` : w)).join(' ')
    const c = await jev().systemOne({ state: { ...state, your_reply: marked }, questions: { pick: choice(`Which option do you pick for the bracketed word "${t.word}"?`, options) } })
    calls++
    const a = c.answers.pick as { choice: string; confidence: number; probabilities: Record<string, number> }
    const top = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).slice(0, 4)
    log(`rewrite [${t.word}]: ${top.map(([o, p]) => `${o.replace(/^word: /, '')} ${(p * 100).toFixed(0)}%`).join(', ')}`)
    if (a.choice === KEEP) continue
    const to = a.choice === DROP ? '' : a.choice.slice(6)
    edits.push({ from: t.word, to, confidence: a.confidence })
    next[t.i] = to
  }
  const after = next.filter(Boolean).join(' ').replace(/ ([.,!?;:…])/g, '$1')
  if (!edits.length || after === answer) return { wants, edits, before: answer, after: answer, ratings: [0, 0], kept: 'before', calls }
  const ratings = (await rateDrafts(question, [answer, after], conversation, mode)) as [number, number]
  calls++
  const kept = ratings[1] > ratings[0] ? 'after' : 'before'
  log(`rewrite: "${answer}" ${(ratings[0] * 100).toFixed(0)}% → "${after}" ${(ratings[1] * 100).toFixed(0)}%, kept ${kept}`)
  return { wants: wants.map(({ word, p }) => ({ word, p })), edits, before: answer, after, ratings, kept, calls }
}
