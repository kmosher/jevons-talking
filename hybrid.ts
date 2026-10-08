// Answers with a single path first, which is cheap and reliable on plain facts, and falls
// back to beam search, which finds better answers to open or self-referential questions,
// only when Jev rates the single-path answer below `good`. Then a fresh call, seeing only the
// question and the candidates, rates the single answer against the beam's best drafts.
//
// Usage: npm run -s talk:hybrid -- [--q="..."] [--good=0.35]
import { type BeamTalk, beam, rateDrafts, scoresFor } from './beam.ts'
import { type Rewrite, rewrite } from './rewrite.ts'
import { SPEAK_GATE, classify, moderate, type Scores, isBare, unfamiliarWords, usefulWords, verdict, relevantContext, repeatedPairs, temperatureFor, type Keyboard, type Mode, type Post, save, talk } from './talk.ts'

const SINGLE_STEPS = 40
// The single path's answer is kept if it rates at least this; otherwise the beam runs.
const HAND_OFF = 0.35
// The salad guard (JEV_SALAD_GUARD=on): below this rating, or above this share of deletes, try the other keyboard.
const SALAD_RATING = 0.3
const SALAD_DELETES = 0.25
// A bare yes or no must rate at least this as a good answer to be kept, at either stage.
const BARE_BAR = 0.6

export type HybridOptions = { keyboard?: Keyboard; keyboardBy?: 'mode' | 'jev'; mode?: Mode; conversation?: Post[]; good?: number; log?: (line: string) => void }
// What was decided before writing started, for the trace image.
export type Decisions = { playful?: number; temperature?: number; mode: Mode; modeConfidence?: number; verdict?: { yes: number; know?: number; word: string }; contextWords?: { word: string; p: number }[]; contextTotal: number; contextKept: number; dropped: Post[]; images: string[] }
export type HybridTalk = BeamTalk & { scores?: Scores; rewrite?: Rewrite; conversation?: Post[]; path: 'single' | 'beam'; mode: Mode; keyboard: Keyboard; decisions: Decisions }

export async function hybrid(question: string, opts: HybridOptions = {}): Promise<HybridTalk> {
  let t: HybridTalk = { ...(await compose(question, opts)), conversation: opts.conversation }
  if (process.env.JEV_REWRITE !== 'off' && t.answer) {
    const rw = await rewrite(question, t.answer, {
      conversation: opts.conversation,
      mode: t.mode,
      extraWords: (t.decisions.contextWords ?? []).filter((w) => w.p >= 0.5).map((w) => w.word),
      locked: t.decisions.verdict ? 1 : 0,
      log: opts.log,
    })
    if (rw) t = { ...t, rewrite: rw, answer: rw.kept === 'after' ? rw.after : t.answer, calls: t.calls + rw.calls }
  }
  if (process.env.JEV_MODERATE !== 'off' && t.answer) {
    // Usually already scored alongside the draft's rating; a separate call only if not.
    const rated = scoresFor(question, t.answer)
    const scores = rated ?? (await moderate(question, t.answer, opts.conversation))
    opts.log?.(`moderation: ${Object.entries(scores).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(', ')}`)
    t = { ...t, scores, calls: t.calls + (rated ? 0 : 1) }
  }
  return t
}

async function compose(question: string, opts: HybridOptions = {}): Promise<HybridTalk> {
  const { good = HAND_OFF, log = () => {} } = opts
  // Thread posts Jev rates irrelevant to the latest message are dropped first (JEV_RELEVANCE=off keeps all).
  let conversation = opts.conversation
  let filterCalls = 0
  let droppedPosts: Post[] = []
  if (conversation && (conversation.length > 3 || conversation.some((p) => p.rejected_drafts?.length)) && process.env.JEV_RELEVANCE !== 'off') {
    const { kept, dropped, trimmed } = await relevantContext(conversation)
    filterCalls = 1
    droppedPosts = dropped
    conversation = kept
    if (trimmed.length) log(`trimmed: ${trimmed.join(', ')}`)
    if (dropped.length) log(`dropped context: ${dropped.map((p) => `${p.author}: "${p.text.slice(0, 50)}"`).join(' | ')}`)
  }
  // First, what kind of reply the message calls for and which keyboard to write it with: by
  // mode (acknowledgements and comebacks on the chat keyboard), or Jev's own choice.
  const keyboardBy = opts.keyboardBy ?? (process.env.JEV_KEYBOARD_BY === 'jev' ? 'jev' : 'mode')
  // Unfamiliar words from the conversation join the word menu: all of them, or (the default)
  // only those Jev says it might use. JEV_CONTEXT_WORDS=all|judge|off.
  const cwMode = process.env.JEV_CONTEXT_WORDS ?? 'judge'
  // JT's own posts are left out, so its "idk," (typed in for it by the verdict) isn't learned back.
  const unfamiliar = cwMode === 'off' ? [] : unfamiliarWords([...(conversation ?? []).filter((p) => p.role !== 'you').map((p) => p.text), question])
  // The mode, verdict and context-word questions share one call (separate calls when the mode is given).
  const classified = opts.mode ? null : await classify(question, conversation, keyboardBy === 'jev' && process.env.JEV_ROUTING === 'on', cwMode === 'judge' ? unfamiliar : [])
  const mode = opts.mode ?? classified!.mode
  // Everything is written on the words keyboard unless JEV_ROUTING=on (keyboard by mode, or by
  // Jev with JEV_KEYBOARD_BY=jev).
  const keyboard: Keyboard = opts.keyboard
    ? opts.keyboard
    : process.env.JEV_ROUTING !== 'on'
      ? 'words'
      : keyboardBy === 'jev' && classified?.keyboard
        ? classified.keyboard
        : mode === 'acknowledge' || mode === 'comeback'
          ? 'chat'
          : 'words'
  log(
    `mode: ${mode}${classified ? ` (${(classified.confidence * 100).toFixed(0)}%)` : ''}, keyboard: ${keyboard}` +
      (classified?.keyboardConfidence !== undefined ? ` (Jev's pick, ${(classified.keyboardConfidence * 100).toFixed(0)}%)` : ''),
  )
  // A yes-or-no question gets Jev's verdict first, typed in for it; JT writes only the reason.
  const v = mode !== 'yesno' ? undefined : (classified?.verdict ?? (await verdict(question, conversation)))
  const prefill = v ? [v.word, ','] : []
  if (v) log(`verdict: ${v.word} (yes ${(v.yes * 100).toFixed(0)}%, knows ${(v.know * 100).toFixed(0)}%)`)
  const contextWords = cwMode !== 'judge' ? unfamiliar.map((word) => ({ word, p: 1 })) : classified ? classified.contextWords : await usefulWords(question, conversation, unfamiliar)
  const extraWords = contextWords.filter((w) => w.p >= 0.5).map((w) => w.word)
  // Asked for something new, JT can't retype its own earlier phrases (JEV_NOVELTY=off skips this).
  const novel = process.env.JEV_NOVELTY !== 'off' && (classified?.novelty ?? 0) >= 0.5
  const banned = novel ? repeatedPairs(conversation) : undefined
  const temperature = temperatureFor(classified?.playful)
  if (classified?.playful !== undefined) log(`playful: ${classified.playful.toFixed(1)} of 5, temperature ${temperature.toFixed(2)}`)
  if (classified?.novelty !== undefined) log(`asks for something new: ${(classified.novelty * 100).toFixed(0)}%${banned ? `; ${banned.size} earlier word pairs off the menu` : ''}`)
  if (unfamiliar.length) log(`context words: ${contextWords.map((w) => `${w.word} ${(w.p * 100).toFixed(0)}%`).join(', ')}`)
  // The single path gets fewer steps: if it hasn't finished by then, the beam is the better bet.
  const write = (k: Keyboard) => talk(question, { keyboard: k, mode, conversation, prefill, extraWords, banned, temperature, maxSteps: SINGLE_STEPS, phrases: Number(process.env.JEV_PHRASES ?? 0), log })
  const drafts = [{ keyboard, t: await write(keyboard) }]
  let ratings = await rateDrafts(question, [drafts[0].t.answer || '…'], conversation, mode)
  // Salad guard: a draft the judge rates poorly, or one that spent a quarter of its picks
  // deleting, gets a second draft on the other keyboard, and the better one goes forward.
  const deletes = drafts[0].t.steps.filter((s) => s.pick === 'backspace').length / (drafts[0].t.steps.length || 1)
  if (process.env.JEV_SALAD_GUARD === 'on' && (ratings[0] < SALAD_RATING || deletes > SALAD_DELETES)) {
    const other: Keyboard = keyboard === 'letters' ? 'words' : 'letters'
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
    drafts.reduce((n, d) => n + d.t.steps.length + d.t.steps.filter((s) => s.final !== undefined).length + d.t.steps.filter((s) => s.complete !== undefined).length + d.t.steps.filter((s) => s.pruned !== undefined).length, 0) + drafts.length + (classified ? 1 : (v ? 1 : 0) + (cwMode === 'judge' && unfamiliar.length ? 1 : 0)) + filterCalls
  log(`single path: "${single.answer}" (${chosenKeyboard}) rated ${(rating * 100).toFixed(0)}% in ${singleCalls} calls`)
  // The question itself is the last conversation entry, so it isn't counted as context.
  const decisions: Decisions = {
    ...(classified?.playful !== undefined ? { playful: classified.playful, temperature } : {}),
    mode,
    modeConfidence: classified?.confidence,
    verdict: v,
    contextWords,
    contextTotal: Math.max(0, (opts.conversation?.length ?? 1) - 1),
    contextKept: Math.max(0, (conversation?.length ?? 1) - 1),
    dropped: droppedPosts,
    images: (conversation ?? []).flatMap((p) => p.images ?? []),
  }
  const asSingle: HybridTalk = {
    decisions,
    ...single,
    drafts: drafts.map((d, i) => ({ answer: d.t.answer, score: ratings[i], picks: d.t.steps.length })),
    judged: Object.fromEntries(drafts.map((d, i) => [d.t.answer, ratings[i]])),
    calls: singleCalls,
    path: 'single',
    mode,
    keyboard: chosenKeyboard,
  }
  // An acknowledgement or comeback of two words or fewer goes to the beam even when it rated
  // well, so the search can get past a first word that SPEAK made too easy to stop at
  // (JEV_EXPLORE=off disables this).
  // With the SPEAK gate on, short fragments are caught while writing, so this is off by default.
  const explore = (process.env.JEV_EXPLORE === 'on' || (process.env.JEV_EXPLORE !== 'off' && !SPEAK_GATE)) && (mode === 'acknowledge' || mode === 'comeback') && single.answer.split(/\s+/).filter((w) => /\w/.test(w)).length <= 2
  if (!explore && rating >= (isBare(single.answer) ? BARE_BAR : good)) return asSingle
  if (explore) log(`exploring past "${single.answer}"`)

  const b = await beam(question, { keyboard: chosenKeyboard, mode, conversation, prefill, extraWords, banned, temperature, seed: drafts.map((d) => d.t.answer).filter(Boolean), log })
  // The final pick is a fresh rating of the single answer and the beam's best drafts.
  const beamBest = Object.entries(b.judged ?? {})
    .filter(([d]) => d !== single.answer && b.draftSteps?.[d])
    .sort((x, y) => y[1] - x[1])
    .slice(0, 3)
    .map(([d]) => d)
  const finalists = [...new Set([...drafts.map((d) => d.t.answer), ...beamBest])].filter(Boolean)
  // Every draft came back empty: nothing to judge, so keep the single path's (empty) result.
  if (!finalists.length) return { ...asSingle, calls: singleCalls + b.calls }
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
  return { ...b, answer: best, steps, judged, calls, path: 'beam', mode, keyboard: chosenKeyboard, decisions }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const t = await hybrid(flag('q') ?? 'What is a black hole?', { good: Number(flag('good') ?? HAND_OFF), log: (l) => console.error(l) })
  console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.path} path, ${t.calls} calls`)
  console.log(`saved ${save(t)}`)
}
