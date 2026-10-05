// Beam search over drafts: every live branch gets its own menu question in a single Jev
// call. A branch splits on each option at or above `split` probability (up to `maxSplit`
// children). The best `width` distinct drafts survive each step, ranked by the geometric
// mean of their picks' probabilities. A branch that picks SPEAK is finished, and also
// continues along its most likely other option.
//
// Each finished draft is rated ("is this a good final answer?") inside the next step's call,
// so rating is free. A draft that only restates the question has its rating cut. The search ends as soon as a draft rates at least `good`, or once the
// finished top `width` outrank every live branch or stop changing; the best-rated draft wins.
//
// Usage: npm run -s talk:beam -- [--q="..."] [--width=3] [--split=0.05] [--max-steps=40] [--scoring=mean|product] [--good=0.8] [--dry-run]
import { choice, noul } from '@typesafe-ai/sdk'
import {
  applyPick,
  type Branch,
  branchKey,
  branchText,
  cloneBranch,
  instructions,
  jev,
  menuFor,
  newBranch,
  type Post,
  type Step,
  recentActions,
  save,
  type Talk,
} from './talk.ts'

export type BeamOptions = {
  // Drafts written elsewhere (the single path's answer) to rate alongside the beam's own.
  seed?: string[]
  conversation?: Post[]
  width?: number
  split?: number
  maxSplit?: number
  maxSteps?: number
  // How branches are ranked: 'mean' (geometric mean of pick probabilities) or 'product'.
  scoring?: 'mean' | 'product'
  // A finished draft rated at least this likely to be a good final answer ends the search.
  good?: number
  dryRun?: boolean
  log?: (line: string) => void
}
export type Draft = { answer: string; score: number; picks: number }
export type BeamTalk = Talk & { drafts: Draft[]; draftSteps?: Record<string, Step[]>; judged: Record<string, number> | null; calls: number }

const beamInstructions = `${instructions}
Several drafts of your answer are being written in parallel; each is a branch under branches_so_far
with its own text and recent actions. Choose each branch's next option independently, as if that
draft were the only one.`
const MENU = { words: 150, common: 30, minWordsToSpeak: 1 }
// A draft that uses no word the question didn't has its rating multiplied by this. Every word
// counts, not just content words: "i am jev" answers "are you Jev?" by its change of person.
export const ECHO_PENALTY = 0.2
const wordsOf = (text: string) => text.toLowerCase().match(/[a-z0-9']+/g) ?? []
// Drafts are rated without trailing punctuation: a final "." swung the same answer's rating
// by 20-30 points ("i like it" 61%, "i like it." 38%).
const forRating = (draft: string) => draft.replace(/[\s.,?!]+$/, '')
const adjust = (draft: string, question: string, rating: number) => (echoes(draft, question) ? rating * ECHO_PENALTY : rating)

// Rates candidate answers in a fresh call that sees only the question, the conversation and
// the candidates: no keyboard instructions, branches or history, which swung ratings of the
// same answer by tens of points.
export async function rateDrafts(question: string, drafts: string[], conversation?: Post[]): Promise<number[]> {
  const ids = drafts.map((_, i) => `a${i}`)
  const r = await jev().systemOne({
    state: {
      ...(conversation?.length ? { conversation_so_far: conversation } : {}),
      question,
      candidate_answers: Object.fromEntries(drafts.map((d, i) => [ids[i], forRating(d)])),
    },
    questions: Object.fromEntries(ids.map((id) => [id, noul(`Is candidate answer \`${id}\` a good answer to the question?`)])),
  })
  return drafts.map((d, i) => adjust(d, question, (r.answers[ids[i]] as { noul: number }).noul))
}

export const echoes = (draft: string, question: string) => {
  const asked = new Set(wordsOf(question))
  return wordsOf(draft).every((w) => asked.has(w) || asked.has(w.replace(/s$/, '')) || asked.has(`${w}s`))
}
// With a full set of finished drafts, stop if it hasn't changed in this many steps.
const STALL_STEPS = 4
const GOOD_ENOUGH = 0.8
const DECENT = 0.4

function distinctBy<T>(items: T[], key: (t: T) => string): T[] {
  const seen = new Set<string>()
  return items.filter((t) => !seen.has(key(t)) && (seen.add(key(t)), true))
}

export async function beam(question: string, opts: BeamOptions = {}): Promise<BeamTalk> {
  const { conversation, seed = [], width = 3, split = 0.05, maxSplit = 3, maxSteps = 40, scoring = 'mean', good = GOOD_ENOUGH, dryRun = false, log = () => {} } = opts
  // A product of probabilities shrinks with every pick, so it favours short drafts; the
  // geometric mean ranks drafts by how confident each pick was, whatever their length.
  const rank = (b: Branch) =>
    scoring === 'product' || !b.steps.length ? b.score : Math.exp(b.steps.reduce((s, st) => s + Math.log(Math.max(st.confidence, 1e-6)), 0) / b.steps.length)
  const byRank = (x: Branch, y: Branch) => rank(y) - rank(x)
  const distinct = (bs: Branch[]) => distinctBy([...bs].sort(byRank), branchText)
  const client = jev()
  let live: Branch[] = [newBranch()]
  const finished: Branch[] = []
  // Each distinct finished draft's yes-probability for "is this a good final answer?".
  const ratings = new Map<string, number>()
  let calls = 0
  let topKey = ''
  let stale = 0

  // Rates `drafts` alongside the step's menu questions, so rating costs no extra calls.
  const ask = async (branchQuestions: Record<string, ReturnType<typeof choice>>, branchState: Record<string, { text_so_far: string; letters_typed: string; recent_actions: string | string[] }> | null, drafts: string[]) => {
    const draftIds = drafts.map((_, i) => `d${i}`)
    const state = {
      instructions: beamInstructions,
      ...(conversation?.length ? { conversation_so_far: conversation } : {}),
      question,
      ...(branchState ? { branches_so_far: branchState } : {}),
      ...(drafts.length ? { finished_drafts: Object.fromEntries(drafts.map((d, i) => [draftIds[i], forRating(d)])) } : {}),
    }
    const questions = {
      ...branchQuestions,
      ...Object.fromEntries(draftIds.map((id) => [id, noul(`Is finished draft \`${id}\` a good final answer to the question?`)])),
    }
    if (dryRun) {
      console.log(JSON.stringify({ state, questions: Object.fromEntries(Object.entries(questions).slice(0, 1)) }, null, 2))
      return null
    }
    const r = await client.systemOne({ state, questions })
    calls++
    drafts.forEach((d, i) => ratings.set(d, adjust(d, question, (r.answers[draftIds[i]] as { noul: number }).noul)))
    return r.answers
  }
  const unrated = () => [...seed, ...distinct(finished).map(branchText)].filter((d, i, all) => d && all.indexOf(d) === i && !ratings.has(d))
  const bestRated = () => Math.max(0, ...ratings.values())

  for (let step = 0; step < maxSteps && live.length; step++) {
    const ids = live.map((_, i) => `b${i}`)
    const menus = live.map((b) => menuFor(b, MENU))
    const branchState = Object.fromEntries(
      live.map((b, i) => [ids[i], { text_so_far: branchText(b) || '(nothing yet)', letters_typed: b.prefix || '(none)', recent_actions: recentActions(b) }]),
    )
    const branchQuestions = Object.fromEntries(ids.map((id, i) => [id, choice(`Which menu option do you pick next for branch \`${id}\`?`, menus[i])]))
    const raw = await ask(branchQuestions, branchState, unrated())
    if (!raw) break
    const answers = raw as Record<string, { choice: string; confidence: number; probabilities: Record<string, number> }>

    const children: Branch[] = []
    live.forEach((b, i) => {
      const a = answers[ids[i]]
      const ranked = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])
      // Always follow Jev's pick; split on other options that clear the bar, except mid-word:
      // splitting on letters filled the beam with spellings of the same word.
      const spelling = b.prefix !== '' || a.choice.startsWith('letter: ')
      const picks = spelling
        ? [a.choice]
        : [a.choice, ...ranked.map(([o]) => o).filter((o) => o !== a.choice && !o.startsWith('letter: ') && a.probabilities[o] >= split)].slice(0, maxSplit)
      // A draft that stops also carries on along its best alternative, so longer answers get written too.
      const goOn = ranked.find(([o]) => o !== 'SPEAK')?.[0]
      if (picks.includes('SPEAK') && goOn && !picks.includes(goOn)) picks.push(goOn)
      for (const pick of picks) {
        const child = cloneBranch(b)
        const p = a.probabilities[pick] ?? a.confidence
        child.steps.push({ menu: Object.keys(menus[i]), pick, confidence: p, probabilities: a.probabilities })
        child.score *= p
        if (pick === 'SPEAK') finished.push(child)
        else {
          applyPick(child, pick)
          children.push(child)
        }
      }
    })

    // Same text and typed letters → same draft; keep the higher-ranked one.
    const byKey = new Map<string, Branch>()
    for (const c of children) if (rank(byKey.get(branchKey(c)) ?? c) <= rank(c)) byKey.set(branchKey(c), c)
    live = [...byKey.values()].sort(byRank).slice(0, width)
    log(
      `${String(step + 1).padStart(2)} ` +
        live.map((b) => `[${rank(b).toFixed(3)}] ${branchText(b)}${b.prefix ? ` ${b.prefix}…` : ''}`).join('  |  ') +
        (ratings.size ? `   rated: ${[...ratings].map(([d, r]) => `"${d}" ${(r * 100).toFixed(0)}%`).join(', ')}` : ''),
    )
    // A draft already rated good enough ends the search.
    if (bestRated() >= good) break
    // Otherwise stop once the finished top `width` outrank every live branch, or stop changing.
    const top = distinct(finished).slice(0, width)
    const key = top.map(branchText).join('\n')
    stale = key === topKey ? stale + 1 : 0
    topKey = key
    // A stall only ends the search once some draft is decent; short junk drafts finish early,
    // and stopping on them cut off longer answers still being written.
    if (top.length >= width && (live.every((b) => rank(b) < rank(top.at(-1)!)) || (stale >= STALL_STEPS && bestRated() >= DECENT))) break
  }

  // Rate whatever finished since the last call, then take the best-rated draft.
  if (!dryRun && unrated().length && bestRated() < good) await ask({}, null, unrated())
  const drafts = distinct(finished)
  const winner = drafts.length
    ? drafts.reduce((best, b) => ((ratings.get(branchText(b)) ?? 0) > (ratings.get(branchText(best)) ?? 0) ? b : best))
    : live[0] ?? newBranch()
  const judged = drafts.length ? Object.fromEntries(drafts.filter((b) => ratings.has(branchText(b))).map((b) => [branchText(b), ratings.get(branchText(b))!])) : null
  if (judged) log(`judge: ${branchText(winner)}  (${Object.entries(judged).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(', ')})`)
  return {
    question,
    answer: branchText(winner),
    finished: drafts.length > 0,
    steps: winner.steps,
    drafts: drafts.map((b) => ({ answer: branchText(b), score: rank(b), picks: b.steps.length })),
    draftSteps: Object.fromEntries(drafts.map((b) => [branchText(b), b.steps])),
    judged,
    calls,
  }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const flag = (name: string) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3)
  const t = await beam(flag('q') ?? 'What is a black hole?', {
    width: Number(flag('width') ?? 3),
    split: Number(flag('split') ?? 0.05),
    maxSteps: Number(flag('max-steps') ?? 40),
    scoring: (flag('scoring') ?? 'mean') as 'mean' | 'product',
    good: Number(flag('good') ?? GOOD_ENOUGH),
    dryRun: args.includes('--dry-run'),
    log: (l) => console.error(l),
  })
  if (!args.includes('--dry-run')) {
    console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.calls} calls; drafts: ${t.drafts.map((d) => `"${d.answer}" ${d.score.toFixed(3)}`).join(', ')}`)
    console.log(`saved ${save(t)}`)
  }
}
