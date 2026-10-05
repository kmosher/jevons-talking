// Beam search over drafts: every live branch gets its own menu question in a single Jev
// call. A branch splits on each option at or above `split` probability (up to `maxSplit`
// children). The best `width` distinct drafts survive each step, ranked by the geometric
// mean of their picks' probabilities. A branch that picks SPEAK is finished, and also
// continues along its most likely other option. Once the finished top `width` outrank
// every live branch, or haven't changed in STALL_STEPS steps, Jev rates each finished draft as a final answer, and the best wins.
//
// Usage: npm run -s talk:beam -- [--q="..."] [--width=3] [--split=0.05] [--max-steps=60] [--scoring=mean|product] [--judge=final|choice] [--dry-run]
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
  recentActions,
  save,
  type Talk,
} from './talk.ts'

export type BeamOptions = {
  conversation?: Post[]
  width?: number
  split?: number
  maxSplit?: number
  maxSteps?: number
  // How branches are ranked: 'mean' (geometric mean of pick probabilities) or 'product'.
  scoring?: 'mean' | 'product'
  // How the winner is chosen: 'final' (a yes/no "good final answer?" per draft) or 'choice' (one head-to-head).
  judge?: 'final' | 'choice'
  dryRun?: boolean
  log?: (line: string) => void
}
export type Draft = { answer: string; score: number; picks: number }
export type BeamTalk = Talk & { drafts: Draft[]; judged: Record<string, number> | null; calls: number }

const beamInstructions = `${instructions}
Several drafts of your answer are being written in parallel; each is a branch under branches_so_far
with its own text and recent actions. Choose each branch's next option independently, as if that
draft were the only one.`
const MENU = { words: 150, common: 30, minWordsToSpeak: 1 }
// With a full set of finished drafts, stop if it hasn't changed in this many steps.
const STALL_STEPS = 8

function distinctBy<T>(items: T[], key: (t: T) => string): T[] {
  const seen = new Set<string>()
  return items.filter((t) => !seen.has(key(t)) && (seen.add(key(t)), true))
}

export async function beam(question: string, opts: BeamOptions = {}): Promise<BeamTalk> {
  const { conversation, width = 3, split = 0.05, maxSplit = 3, maxSteps = 60, scoring = 'mean', judge = 'final', dryRun = false, log = () => {} } = opts
  // A product of probabilities shrinks with every pick, so it favours short drafts; the
  // geometric mean ranks drafts by how confident each pick was, whatever their length.
  const rank = (b: Branch) =>
    scoring === 'product' || !b.steps.length ? b.score : Math.exp(b.steps.reduce((s, st) => s + Math.log(Math.max(st.confidence, 1e-6)), 0) / b.steps.length)
  const byRank = (x: Branch, y: Branch) => rank(y) - rank(x)
  const distinct = (bs: Branch[]) => distinctBy([...bs].sort(byRank), branchText)
  const client = jev()
  let live: Branch[] = [newBranch()]
  const finished: Branch[] = []
  let calls = 0
  // Steps since the finished top `width` last changed.
  let topKey = ''
  let stale = 0

  for (let step = 0; step < maxSteps && live.length; step++) {
    const ids = live.map((_, i) => `b${i}`)
    const menus = live.map((b) => menuFor(b, MENU))
    const state = {
      instructions: beamInstructions,
      ...(conversation?.length ? { conversation_so_far: conversation } : {}),
      question,
      branches_so_far: Object.fromEntries(
        live.map((b, i) => [ids[i], { text_so_far: branchText(b) || '(nothing yet)', letters_typed: b.prefix || '(none)', recent_actions: recentActions(b) }]),
      ),
    }
    const questions = Object.fromEntries(ids.map((id, i) => [id, choice(`Which menu option do you pick next for branch \`${id}\`?`, menus[i])]))
    if (dryRun) {
      console.log(JSON.stringify({ state, questions: { [ids[0]]: questions[ids[0]] } }, null, 2))
      break
    }
    const r = await client.systemOne({ state, questions })
    calls++
    const answers = r.answers as Record<string, { choice: string; confidence: number; probabilities: Record<string, number> }>

    const children: Branch[] = []
    live.forEach((b, i) => {
      const a = answers[ids[i]]
      const ranked = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])
      // Always follow Jev's pick; split on other options that clear the bar.
      const picks = [a.choice, ...ranked.map(([o]) => o).filter((o) => o !== a.choice && a.probabilities[o] >= split)].slice(0, maxSplit)
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

    // Same text and typed letters → same draft; keep the higher-scoring one.
    const byKey = new Map<string, Branch>()
    for (const c of children) if (rank(byKey.get(branchKey(c)) ?? c) <= rank(c)) byKey.set(branchKey(c), c)
    live = [...byKey.values()].sort(byRank).slice(0, width)
    const bestFinished = Math.max(0, ...finished.map(rank))
    log(
      `${String(step + 1).padStart(2)} ` +
        live.map((b) => `[${rank(b).toFixed(3)}] ${branchText(b)}${b.prefix ? ` ${b.prefix}…` : ''}`).join('  |  ') +
        (finished.length ? `   ✓${finished.length} best ${bestFinished.toFixed(3)}` : ''),
    )
    // Stop once the finished top `width` all outrank every live branch. (Under product scoring
    // that's exact, as scores only fall; under mean scoring it's a heuristic.)
    const top = distinct(finished).slice(0, width)
    const key = top.map(branchText).join('\n')
    stale = key === topKey ? stale + 1 : 0
    topKey = key
    if (top.length >= width && (live.every((b) => rank(b) < rank(top.at(-1)!)) || stale >= STALL_STEPS)) break
  }

  // Distinct finished drafts, best first; fall back to the best live draft.
  const finishedDrafts = distinct(finished)
  const pool = finishedDrafts.length ? finishedDrafts.slice(0, width) : live.slice(0, 1)
  let winner = pool[0]
  let judged: Record<string, number> | null = null
  if (pool.length > 1 && !dryRun && judge === 'final') {
    const ids = pool.map((_, i) => `d${i}`)
    const j = await client.systemOne({
      state: {
        ...(conversation?.length ? { conversation_so_far: conversation } : {}),
        question,
        drafts: Object.fromEntries(pool.map((b, i) => [ids[i], branchText(b)])),
      },
      questions: Object.fromEntries(ids.map((id) => [id, noul(`Is draft \`${id}\` a good final answer to the question?`)])),
    })
    calls++
    judged = Object.fromEntries(pool.map((b, i) => [branchText(b), (j.answers[ids[i]] as { noul: number }).noul]))
    winner = pool.reduce((best, b) => (judged![branchText(b)] > judged![branchText(best)] ? b : best))
    log(`judge: ${branchText(winner)}  (${Object.entries(judged).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(', ')})`)
  } else if (pool.length > 1 && !dryRun) {
    const options = Object.fromEntries(pool.map((b) => [branchText(b), `the answer "${branchText(b)}"`]))
    const j = await client.systemOne({
      state: { ...(conversation?.length ? { conversation_so_far: conversation } : {}), question },
      questions: { best: choice('Which of these drafts is the best answer to the question?', options) },
    })
    calls++
    const a = j.answers.best as { choice: string; probabilities: Record<string, number> }
    judged = a.probabilities
    winner = pool.find((b) => branchText(b) === a.choice) ?? winner
    log(`judge: ${a.choice}  (${Object.entries(a.probabilities).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(', ')})`)
  }
  return {
    question,
    answer: branchText(winner),
    finished: finishedDrafts.length > 0,
    steps: winner.steps,
    drafts: pool.map((b) => ({ answer: branchText(b), score: rank(b), picks: b.steps.length })),
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
    maxSteps: Number(flag('max-steps') ?? 60),
    scoring: (flag('scoring') ?? 'mean') as 'mean' | 'product',
    judge: (flag('judge') ?? 'final') as 'final' | 'choice',
    dryRun: args.includes('--dry-run'),
    log: (l) => console.error(l),
  })
  if (!args.includes('--dry-run')) {
    console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.calls} calls; drafts: ${t.drafts.map((d) => `"${d.answer}" ${d.score.toFixed(3)}`).join(', ')}`)
    console.log(`saved ${save(t)}`)
  }
}
