// Beam search over drafts: every live branch gets its own menu question in a single Jev
// call. A branch splits on each option at or above SPLIT probability (up to MAX_SPLIT
// children), scoring children by the product of their picks' probabilities. The best
// `width` distinct drafts survive each step; a branch that picks SPEAK is finished, and
// also continues along its most likely other option.
// Once no live branch can outscore the finished top `width` drafts, Jev picks the best of
// those in one more choice question. Scores only decide which branches survive, never the winner: a probability
// product favours short answers, which is the bias this is meant to escape.
//
// Usage: npm run -s talk:beam -- [--q="..."] [--width=3] [--split=0.05] [--max-steps=60] [--dry-run]
import { choice } from '@typesafe-ai/sdk'
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

export async function beam(question: string, opts: BeamOptions = {}): Promise<BeamTalk> {
  const { conversation, width = 3, split = 0.05, maxSplit = 3, maxSteps = 60, dryRun = false, log = () => {} } = opts
  const client = jev()
  let live: Branch[] = [newBranch()]
  const finished: Branch[] = []
  let calls = 0

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
    for (const c of children) if ((byKey.get(branchKey(c))?.score ?? -1) < c.score) byKey.set(branchKey(c), c)
    live = [...byKey.values()].sort((x, y) => y.score - x.score).slice(0, width)
    const bestFinished = Math.max(0, ...finished.map((f) => f.score))
    log(
      `${String(step + 1).padStart(2)} ` +
        live.map((b) => `[${b.score.toFixed(3)}] ${branchText(b)}${b.prefix ? ` ${b.prefix}…` : ''}`).join('  |  ') +
        (finished.length ? `   ✓${finished.length} best ${bestFinished.toFixed(3)}` : ''),
    )
    // Stop once no live branch can still make the finished top `width` (scores only fall).
    const top = [...new Map([...finished].sort((x, y) => y.score - x.score).reverse().map((f) => [branchText(f), f])).values()]
      .sort((x, y) => y.score - x.score)
      .slice(0, width)
    if (top.length >= width && live.every((b) => b.score < top.at(-1)!.score)) break
  }

  // Distinct finished drafts, best score first; fall back to the best live draft.
  const drafts = new Map<string, Branch>()
  for (const f of [...finished].sort((x, y) => y.score - x.score)) if (!drafts.has(branchText(f))) drafts.set(branchText(f), f)
  const pool = drafts.size ? [...drafts.values()].slice(0, width) : live.slice(0, 1)
  let winner = pool[0]
  let judged: Record<string, number> | null = null
  if (pool.length > 1 && !dryRun) {
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
    finished: drafts.size > 0,
    steps: winner.steps,
    drafts: pool.map((b) => ({ answer: branchText(b), score: b.score, picks: b.steps.length })),
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
    dryRun: args.includes('--dry-run'),
    log: (l) => console.error(l),
  })
  if (!args.includes('--dry-run')) {
    console.log(`\nQ: ${t.question}\nA: ${t.answer}\n${t.calls} calls; drafts: ${t.drafts.map((d) => `"${d.answer}" ${d.score.toFixed(3)}`).join(', ')}`)
    console.log(`saved ${save(t)}`)
  }
}
