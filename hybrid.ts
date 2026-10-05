// Answers with a single path first, which is cheap and reliable on plain facts, and falls
// back to beam search, which finds better answers to open or self-referential questions,
// only when Jev rates the single-path answer below `good`.
//
// Usage: npm run -s talk:hybrid -- [--q="..."] [--good=0.5]
import { noul } from '@typesafe-ai/sdk'
import { type BeamTalk, beam, ECHO_PENALTY, echoes } from './beam.ts'
import { jev, type Post, save, talk } from './talk.ts'

const SINGLE_STEPS = 40

export type HybridOptions = { conversation?: Post[]; good?: number; log?: (line: string) => void }
export type HybridTalk = BeamTalk & { path: 'single' | 'beam' }

// "Is this a good final answer?" as a yes-probability, cut for drafts that restate the question.
async function rate(question: string, answer: string, conversation?: Post[]): Promise<number> {
  const r = await jev().systemOne({
    state: { ...(conversation?.length ? { conversation_so_far: conversation } : {}), question, draft: answer },
    questions: { good: noul('Is the draft a good final answer to the question?') },
  })
  const rating = (r.answers.good as { noul: number }).noul
  return echoes(answer, question) ? rating * ECHO_PENALTY : rating
}

export async function hybrid(question: string, opts: HybridOptions = {}): Promise<HybridTalk> {
  const { conversation, good = 0.5, log = () => {} } = opts
  // The single path gets fewer steps: if it hasn't finished by then, the beam is the better bet.
  const single = await talk(question, { conversation, maxSteps: SINGLE_STEPS, log })
  // One call per pick, plus one per final-answer check, plus the rating below.
  const singleCalls = single.steps.length + single.steps.filter((s) => s.final !== undefined).length + 1
  const rating = single.answer ? await rate(question, single.answer, conversation) : 0
  log(`single path: "${single.answer}" rated ${(rating * 100).toFixed(0)}% in ${singleCalls} calls`)
  const asSingle: HybridTalk = {
    ...single,
    drafts: [{ answer: single.answer, score: rating, picks: single.steps.length }],
    judged: { [single.answer]: rating },
    calls: singleCalls,
    path: 'single',
  }
  if (rating >= good) return asSingle

  const b = await beam(question, { conversation, log })
  const beamRating = b.judged?.[b.answer] ?? 0
  log(`beam: "${b.answer}" rated ${(beamRating * 100).toFixed(0)}% in ${b.calls} calls`)
  const calls = singleCalls + b.calls
  const judged = { ...b.judged, [single.answer]: rating }
  return beamRating > rating ? { ...b, judged, calls, path: 'beam' } : { ...asSingle, judged, calls }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const t = await hybrid(flag('q') ?? 'What is a black hole?', { good: Number(flag('good') ?? 0.5), log: (l) => console.error(l) })
  console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.path} path, ${t.calls} calls`)
  console.log(`saved ${save(t)}`)
}
