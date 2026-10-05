// A Dasher-style keyboard: a character model trained on Dasher's English training text
// (JEV_DASHER_TRAINING) ranks what could come next. Each menu offers every single character
// plus the most likely chunks that finish the current word, each with its probability, the
// way Dasher draws likelier continuations as bigger boxes.
import { readFileSync } from 'node:fs'

const ORDER = 5
const CHARS = "abcdefghijklmnopqrstuvwxyz' .,?"
const COMPLETIONS = 100
const MAX_CHUNK = 14

// counts[k] maps a k-character context to the counts of the character that followed it.
const counts: Map<string, Map<string, number>>[] = Array.from({ length: ORDER + 1 }, () => new Map())
let trained = false
function train() {
  if (trained) return
  trained = true
  const path = process.env.JEV_DASHER_TRAINING
  if (!path) throw new Error('JEV_DASHER_TRAINING is not set')
  const text = ` ${readFileSync(path, 'utf8').toLowerCase().replace(/\s+/g, ' ')}`
  for (let i = 1; i < text.length; i++) {
    const c = text[i]
    if (!CHARS.includes(c)) continue
    for (let k = 0; k <= ORDER && k <= i; k++) {
      const ctx = text.slice(i - k, i)
      const m = counts[k].get(ctx) ?? new Map<string, number>()
      m.set(c, (m.get(c) ?? 0) + 1)
      counts[k].set(ctx, m)
    }
  }
}

// P(next character | context), interpolating every order from 0 up (Witten-Bell weights).
function nextDist(context: string): Map<string, number> {
  const dist = new Map<string, number>(CHARS.split('').map((c) => [c, 1 / CHARS.length]))
  for (let k = 0; k <= ORDER; k++) {
    if (context.length < k) break
    const m = counts[k].get(context.slice(context.length - k))
    if (!m) continue
    const total = [...m.values()].reduce((a, b) => a + b, 0)
    const lambda = total / (total + m.size)
    for (const c of CHARS) dist.set(c, (1 - lambda) * dist.get(c)! + (lambda * (m.get(c) ?? 0)) / total)
  }
  return dist
}

// The menu for text typed so far: every character, plus the likeliest word-finishing chunks.
export function dasherMenu(typed: string): Record<string, string> {
  train()
  const context = ` ${typed}`
  const menu: Record<string, string> = {}
  const first = nextDist(context)
  for (const c of CHARS) {
    const label = c === ' ' ? 'space' : c
    menu[`type: ${label}`] = `type "${label}" (${(first.get(c)! * 100).toFixed(1)}%)`
  }
  // Best-first search for chunks that end the current word with a space or punctuation.
  type Path = { s: string; p: number }
  let frontier: Path[] = [{ s: '', p: 1 }]
  const done: Path[] = []
  while (frontier.length && done.length < COMPLETIONS * 3) {
    frontier.sort((a, b) => b.p - a.p)
    const { s, p } = frontier.shift()!
    for (const [c, q] of nextDist(context + s)) {
      const next = { s: s + c, p: p * q }
      if (next.p < 1e-5) continue
      if (/[ .,?]$/.test(next.s)) {
        if (next.s.length > 2) done.push(next)
      } else if (next.s.length < MAX_CHUNK) frontier.push(next)
    }
    frontier = frontier.sort((a, b) => b.p - a.p).slice(0, 400)
  }
  for (const { s, p } of done.sort((a, b) => b.p - a.p).slice(0, COMPLETIONS)) {
    const shown = s.replace(/ $/, '␣')
    menu[`type: ${shown}`] = `type "${shown}" (${(p * 100).toFixed(1)}%)`
  }
  return menu
}

// The characters a "type:" option stands for.
export const chunkOf = (option: string) => option.slice(6).replace('␣', ' ').replace(/^space$/, ' ')
