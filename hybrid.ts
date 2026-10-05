// Answers with a single path first, which is cheap and reliable on plain facts, and falls
// back to beam search, which finds better answers to open or self-referential questions,
// only when Jev rates the single-path answer below `good`. Then a fresh call, seeing only the
// question and the candidates, rates the single answer against the beam's best drafts.
//
// Usage: npm run -s talk:hybrid -- [--q="..."] [--good=0.35]
import { type BeamTalk, beam, rateDrafts } from './beam.ts'
import { isBare, type Post, save, talk } from './talk.ts'

const SINGLE_STEPS = 40
// The single path's answer is kept if it rates at least this; otherwise the beam runs.
const HAND_OFF = 0.35
// A bare yes or no must rate at least this as a good answer to be kept, at either stage.
const BARE_BAR = 0.6

export type HybridOptions = { conversation?: Post[]; good?: number; log?: (line: string) => void }
export type HybridTalk = BeamTalk & { path: 'single' | 'beam' }

export async function hybrid(question: string, opts: HybridOptions = {}): Promise<HybridTalk> {
  const { conversation, good = HAND_OFF, log = () => {} } = opts
  // The single path gets fewer steps: if it hasn't finished by then, the beam is the better bet.
  const single = await talk(question, { conversation, maxSteps: SINGLE_STEPS, log })
  // One call per pick, plus one per final-answer check, plus the rating below.
  const singleCalls = single.steps.length + single.steps.filter((s) => s.final !== undefined).length + 1
  const rating = single.answer ? (await rateDrafts(question, [single.answer], conversation))[0] : 0
  log(`single path: "${single.answer}" rated ${(rating * 100).toFixed(0)}% in ${singleCalls} calls`)
  const asSingle: HybridTalk = {
    ...single,
    drafts: [{ answer: single.answer, score: rating, picks: single.steps.length }],
    judged: { [single.answer]: rating },
    calls: singleCalls,
    path: 'single',
  }
  if (rating >= (isBare(single.answer) ? BARE_BAR : good)) return asSingle

  const b = await beam(question, { conversation, seed: [single.answer], log })
  // The final pick is a fresh rating of the single answer and the beam's best drafts.
  const beamBest = Object.entries(b.judged ?? {})
    .filter(([d]) => d !== single.answer && b.draftSteps?.[d])
    .sort((x, y) => y[1] - x[1])
    .slice(0, 3)
    .map(([d]) => d)
  const finalists = [...new Set([single.answer, ...beamBest])].filter(Boolean)
  const fresh = await rateDrafts(question, finalists, conversation)
  const judged = Object.fromEntries(finalists.map((d, i) => [d, fresh[i]]))
  // Bare yes/no candidates under BARE_BAR are out, unless nothing else is left.
  const eligible = fresh.map((r, i) => (isBare(finalists[i]) && r < BARE_BAR ? -1 : r))
  const scores = eligible.some((r) => r >= 0) ? eligible : fresh
  const best = finalists[scores.indexOf(Math.max(...scores))]
  log(`final pick: "${best}" (${finalists.map((d, i) => `"${d}" ${(fresh[i] * 100).toFixed(0)}%`).join(', ')})`)
  const calls = singleCalls + b.calls + 1
  if (best === single.answer) return { ...asSingle, judged, calls }
  // The beam's answer may be a draft other than its own pick; use that draft's path for the trace.
  const steps = b.answer === best ? b.steps : (b.draftSteps?.[best] ?? b.steps)
  return { ...b, answer: best, steps, judged, calls, path: 'beam' }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const t = await hybrid(flag('q') ?? 'What is a black hole?', { good: Number(flag('good') ?? HAND_OFF), log: (l) => console.error(l) })
  console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.path} path, ${t.calls} calls`)
  console.log(`saved ${save(t)}`)
}
