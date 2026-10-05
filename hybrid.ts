// Answers with a single path first, which is cheap and reliable on plain facts, and falls
// back to beam search, which finds better answers to open or self-referential questions,
// only when Jev rates the single-path answer below `good`. Then a fresh call, seeing only the
// question and the candidates, rates the single answer against the beam's best drafts.
//
// Usage: npm run -s talk:hybrid -- [--q="..."] [--good=0.35]
import { type BeamTalk, beam, rateDrafts } from './beam.ts'
import { classify, isBare, type Keyboard, type Mode, type Post, save, talk } from './talk.ts'

const SINGLE_STEPS = 40
// The single path's answer is kept if it rates at least this; otherwise the beam runs.
const HAND_OFF = 0.35
// The salad guard: below this rating, or above this share of deletes, try the other keyboard.
const SALAD_RATING = 0.3
const SALAD_DELETES = 0.25
// A bare yes or no must rate at least this as a good answer to be kept, at either stage.
const BARE_BAR = 0.6

export type HybridOptions = { keyboardBy?: 'mode' | 'jev'; mode?: Mode; conversation?: Post[]; good?: number; log?: (line: string) => void }
export type HybridTalk = BeamTalk & { path: 'single' | 'beam'; mode: Mode; keyboard: Keyboard }

export async function hybrid(question: string, opts: HybridOptions = {}): Promise<HybridTalk> {
  const { conversation, good = HAND_OFF, log = () => {} } = opts
  // First, what kind of reply the message calls for and which keyboard to write it with: by
  // mode (acknowledgements and comebacks on the plain keyboard), or Jev's own choice.
  const keyboardBy = opts.keyboardBy ?? (process.env.JEV_KEYBOARD_BY === 'jev' ? 'jev' : 'mode')
  const classified = opts.mode ? null : await classify(question, conversation, keyboardBy === 'jev')
  const mode = opts.mode ?? classified!.mode
  // JEV_ROUTING=off keeps everything on the words keyboard.
  const keyboard: Keyboard =
    process.env.JEV_ROUTING === 'off'
      ? 'words'
      : keyboardBy === 'jev' && classified?.keyboard
        ? classified.keyboard
        : mode === 'acknowledge' || mode === 'comeback'
          ? 'letters'
          : 'words'
  log(
    `mode: ${mode}${classified ? ` (${(classified.confidence * 100).toFixed(0)}%)` : ''}, keyboard: ${keyboard}` +
      (classified?.keyboardConfidence !== undefined ? ` (Jev's pick, ${(classified.keyboardConfidence * 100).toFixed(0)}%)` : ''),
  )
  // The single path gets fewer steps: if it hasn't finished by then, the beam is the better bet.
  const write = (k: Keyboard) => talk(question, { keyboard: k, mode, conversation, maxSteps: SINGLE_STEPS, phrases: Number(process.env.JEV_PHRASES ?? 0), log })
  const drafts = [{ keyboard, t: await write(keyboard) }]
  let ratings = await rateDrafts(question, [drafts[0].t.answer || '…'], conversation, mode)
  // Salad guard: a draft the judge rates poorly, or one that spent a quarter of its picks
  // deleting, gets a second draft on the other keyboard, and the better one goes forward.
  const deletes = drafts[0].t.steps.filter((s) => s.pick === 'backspace').length / (drafts[0].t.steps.length || 1)
  if (process.env.JEV_SALAD_GUARD !== 'off' && (ratings[0] < SALAD_RATING || deletes > SALAD_DELETES)) {
    const other: Keyboard = keyboard === 'words' ? 'letters' : 'words'
    log(`salad guard: "${drafts[0].t.answer}" rated ${(ratings[0] * 100).toFixed(0)}%, ${(deletes * 100).toFixed(0)}% deletes; trying ${other}`)
    drafts.push({ keyboard: other, t: await write(other) })
    ratings = await rateDrafts(question, drafts.map((d) => d.t.answer || '…'), conversation, mode)
  }
  const pick = ratings.indexOf(Math.max(...ratings))
  const single = drafts[pick].t
  const chosenKeyboard = drafts[pick].keyboard
  const rating = single.answer ? ratings[pick] : 0
  // One call per pick and per final-answer check, plus the rating calls and the classification.
  const singleCalls =
    drafts.reduce((n, d) => n + d.t.steps.length + d.t.steps.filter((s) => s.final !== undefined).length, 0) + drafts.length + (classified ? 1 : 0)
  log(`single path: "${single.answer}" (${chosenKeyboard}) rated ${(rating * 100).toFixed(0)}% in ${singleCalls} calls`)
  const asSingle: HybridTalk = {
    ...single,
    drafts: drafts.map((d, i) => ({ answer: d.t.answer, score: ratings[i], picks: d.t.steps.length })),
    judged: Object.fromEntries(drafts.map((d, i) => [d.t.answer, ratings[i]])),
    calls: singleCalls,
    path: 'single',
    mode,
    keyboard: chosenKeyboard,
  }
  // With JEV_EXPLORE=1, a reply of two words or fewer goes to the beam even when it rated well,
  // so the search can get past a first word that SPEAK made too easy to stop at.
  const explore = process.env.JEV_EXPLORE === '1' && single.answer.split(/\s+/).filter((w) => /\w/.test(w)).length <= 2
  if (!explore && rating >= (isBare(single.answer) ? BARE_BAR : good)) return asSingle
  if (explore) log(`exploring past "${single.answer}"`)

  const b = await beam(question, { keyboard: chosenKeyboard, mode, conversation, seed: drafts.map((d) => d.t.answer).filter(Boolean), log })
  // The final pick is a fresh rating of the single answer and the beam's best drafts.
  const beamBest = Object.entries(b.judged ?? {})
    .filter(([d]) => d !== single.answer && b.draftSteps?.[d])
    .sort((x, y) => y[1] - x[1])
    .slice(0, 3)
    .map(([d]) => d)
  const finalists = [...new Set([...drafts.map((d) => d.t.answer), ...beamBest])].filter(Boolean)
  const fresh = await rateDrafts(question, finalists, conversation, mode)
  const judged = Object.fromEntries(finalists.map((d, i) => [d, fresh[i]]))
  // Bare yes/no candidates under BARE_BAR are out, unless nothing else is left.
  const eligible = fresh.map((r, i) => (isBare(finalists[i]) && r < BARE_BAR ? -1 : r))
  const scores = eligible.some((r) => r >= 0) ? eligible : fresh
  const best = finalists[scores.indexOf(Math.max(...scores))]
  log(`final pick: "${best}" (${finalists.map((d, i) => `"${d}" ${(fresh[i] * 100).toFixed(0)}%`).join(', ')})`)
  const calls = singleCalls + b.calls + 1
  const fromSingle = drafts.findIndex((d) => d.t.answer === best)
  if (fromSingle >= 0) return { ...asSingle, ...drafts[fromSingle].t, keyboard: drafts[fromSingle].keyboard, judged, calls }
  // The beam's answer may be a draft other than its own pick; use that draft's path for the trace.
  const steps = b.answer === best ? b.steps : (b.draftSteps?.[best] ?? b.steps)
  return { ...b, answer: best, steps, judged, calls, path: 'beam', mode, keyboard: chosenKeyboard }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const t = await hybrid(flag('q') ?? 'What is a black hole?', { good: Number(flag('good') ?? HAND_OFF), log: (l) => console.error(l) })
  console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.path} path, ${t.calls} calls`)
  console.log(`saved ${save(t)}`)
}
