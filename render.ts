// Renders a transcript as a PNG: the question and answer, the drafts a beam search rated, and
// every keystroke Jev made with a bar for how confident it was. With `full`, it adds a table
// with one row per pick: the options passed over and the text as it stood afterwards.
//
// Usage: node --import tsx render.ts <transcript.json> [out.png] [--full]
import { readFileSync, writeFileSync } from 'node:fs'
import { Resvg } from '@resvg/resvg-js'
import type { BeamTalk } from './beam.ts'
import type { HybridTalk } from './hybrid.ts'
import { meanConfidence, type Step, type Talk } from './talk.ts'

const W = 1080
const PAD = 40
const ROW = 36
const MAX_ROWS = 60
const ALTERNATIVES = 4
const C = { bg: '#f7f4ee', ink: '#1f3a5f', muted: '#8a94a3', orange: '#e8743b', chip: '#dce1e8', line: '#e2ddd2', red: '#c2453d' }
const SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const SERIF = "Georgia, 'Times New Roman', serif"
const MONO = 'Menlo, Monaco, monospace'

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)
// Rough advance width; good enough for sizing chips.
const textWidth = (s: string, size: number) => [...s].reduce((w, ch) => w + (/[A-Z]/.test(ch) ? 0.72 : 0.56), 0) * size

const label = (option: string) => (option === 'backspace' ? '⌫ delete' : option.replace(/^(word|letter): /, ''))
// A SPEAK that Jev then said wasn't final.
const withdrawn = (step: Step) => step.pick === 'SPEAK' && step.final !== undefined && step.final < 0.5
const kind = (option: string) =>
  option === 'SPEAK' ? 'speak' : option === 'backspace' ? 'backspace' : option.startsWith('letter: ') ? 'letter' : 'word'

// The text (and any letters typed toward the next word) after each step, replaying talk()'s editing rules.
function replay(steps: Step[]): { text: string; prefix: string }[] {
  const words: string[] = []
  let prefix = ''
  return steps.map((s) => {
    if (s.pick === 'backspace') {
      if (prefix) prefix = prefix.slice(0, -1)
      else words.pop()
    } else if (s.pick.startsWith('letter: ')) prefix += s.pick.slice(8)
    else if (s.pick.startsWith('word: ')) {
      words.push(s.pick.slice(6))
      prefix = ''
    } else if (s.pick !== 'SPEAK') words.push(s.pick)
    return { text: words.join(' ').replace(/ ([.,?])/g, '$1'), prefix }
  })
}

function chip(x: number, y: number, text: string, opts: { fill: string; ink: string; stroke?: string; size?: number; bold?: boolean; family?: string; opacity?: number }) {
  const size = opts.size ?? 15
  const w = textWidth(text, size) + 18
  const stroke = opts.stroke ? ` stroke="${opts.stroke}" stroke-width="2"` : ''
  return {
    w,
    svg: `<g opacity="${opts.opacity ?? 1}"><rect x="${x}" y="${y}" width="${w}" height="26" rx="7" fill="${opts.fill}"${stroke}/>` +
      `<text x="${x + w / 2}" y="${y + 18}" font-family="${opts.family ?? SANS}" font-size="${size}" font-weight="${opts.bold ? 700 : 400}" fill="${opts.ink}" text-anchor="middle">${esc(text)}</text></g>`,
  }
}

function row(step: Step, i: number, after: { text: string; prefix: string }, y: number): string {
  const parts: string[] = []
  parts.push(`<text x="${PAD}" y="${y + 18}" font-family="${MONO}" font-size="13" fill="${C.muted}" text-anchor="start">${i + 1}</text>`)

  // The pick, styled by kind, with its confidence underneath as a thin bar.
  const k = kind(step.pick)
  const style = {
    word: { fill: C.orange, ink: '#fff', bold: true },
    letter: { fill: '#fff', ink: C.orange, stroke: C.orange, bold: true, family: MONO },
    backspace: { fill: '#fff', ink: C.red, stroke: C.red, bold: true },
    speak: { fill: C.ink, ink: '#fff', bold: true },
  }[k]
  const pick = chip(PAD + 36, y, clip(withdrawn(step) ? `final? ${Math.round(step.final! * 100)}% ✗` : label(step.pick), 18), withdrawn(step) ? { fill: '#fff', ink: C.ink, stroke: C.ink, bold: true } : style)
  parts.push(pick.svg)
  parts.push(`<rect x="${PAD + 36}" y="${y + 29}" width="${pick.w}" height="3" rx="1.5" fill="${C.line}"/>`)
  parts.push(`<rect x="${PAD + 36}" y="${y + 29}" width="${pick.w * step.confidence}" height="3" rx="1.5" fill="${C.ink}"/>`)

  // Runner-up options by probability.
  let x = PAD + 236
  const alts = Object.entries(step.probabilities)
    .filter(([o]) => o !== step.pick)
    .sort((a, b) => b[1] - a[1])
    .slice(0, ALTERNATIVES)
  for (const [option, p] of alts) {
    const c = chip(x, y, `${clip(label(option), 12)} ${Math.round(p * 100)}%`, { fill: C.chip, ink: C.ink, size: 13, opacity: 0.45 + 0.55 * Math.min(1, p / 0.2) })
    if (x + c.w > W - PAD - 300) break
    parts.push(c.svg)
    x += c.w + 6
  }

  // The running text: settled words in ink, letters typed so far in orange.
  const tail = clip([...after.text].reverse().join(''), 34)
  const shown = [...tail].reverse().join('')
  parts.push(
    `<text x="${W - PAD}" y="${y + 19}" font-family="${SERIF}" font-size="17" fill="${C.ink}" text-anchor="end">${esc(shown)}` +
      (after.prefix ? `<tspan fill="${C.orange}" font-weight="700">${after.text ? '\u00a0' : ''}${esc(after.prefix)}▁</tspan>` : '') +
      '</text>',
  )
  return parts.join('')
}

export function renderSvg(t: Talk, { full = false } = {}): string {
  const states = replay(t.steps)
  let indices = t.steps.map((_, i) => i)
  let gapAfter = -1
  if (indices.length > MAX_ROWS) {
    const head = Math.floor(MAX_ROWS / 2)
    gapAfter = head - 1
    indices = [...indices.slice(0, head), ...indices.slice(-(MAX_ROWS - head))]
  }

  const top = PAD
  const parts: string[] = []
  parts.push(`<text x="${PAD}" y="${top + 16}" font-family="${SANS}" font-size="15" fill="${C.muted}">Q: ${esc(clip(t.question, 100))}</text>`)
  parts.push(`<text x="${PAD}" y="${top + 58}" font-family="${SERIF}" font-size="34" font-weight="700" fill="${C.ink}">${esc(clip(t.answer || '…', 56))}</text>`)
  const calls = (t as Partial<BeamTalk>).calls
  const meta = `${t.steps.length} picks · mean confidence ${meanConfidence(t).toFixed(2)}${calls ? ` · ${calls} Jev calls` : ''}${t.finished ? '' : ' · ran out of picks'}`
  parts.push(`<text x="${PAD}" y="${top + 88}" font-family="${SANS}" font-size="14" fill="${C.muted}">${esc(meta)}</text>`)

  let y = top + 112
  // The decisions made before writing: reply mode, how much context was kept, what images it
  // "saw", and the path. With full, the dropped posts and complete image descriptions too.
  const d = (t as Partial<HybridTalk>).decisions
  if (d) {
    const path = (t as Partial<HybridTalk>).path
    const bits = [
      `mode: ${d.mode}${d.modeConfidence !== undefined ? ` ${Math.round(d.modeConfidence * 100)}%` : ''}`,
      ...(d.contextTotal ? [`context: kept ${d.contextKept} of ${d.contextTotal} posts`] : []),
      ...d.images.map((img) => `saw: "${clip(img, full ? 200 : 60)}"`),
      ...(path ? [`path: ${path === 'beam' ? `beam → ${Object.keys((t as Partial<HybridTalk>).judged ?? {}).length} drafts` : 'single draft'}`] : []),
    ]
    parts.push(`<text x="${PAD}" y="${y - 2}" font-family="${SANS}" font-size="14" fill="${C.ink}">${esc(bits.join('  ·  '))}</text>`)
    y += 26
    if (full)
      for (const p of d.dropped) {
        parts.push(`<text x="${PAD}" y="${y - 2}" font-family="${SANS}" font-size="13" fill="${C.muted}">${esc(`dropped: ${p.author}: "${clip(p.text.replace(/\s+/g, ' '), 110)}"`)}</text>`)
        y += 20
      }
    y += 6
  }
  // When there was more than one candidate: each, and how Jev rated it as a final answer.
  const judged = (t as Partial<BeamTalk>).judged
  if (judged && Object.keys(judged).length > 1) {
    parts.push(`<text x="${PAD}" y="${y + 4}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">DRAFTS · RATED AS A FINAL ANSWER</text>`)
    y += 26
    for (const [draft, p] of Object.entries(judged).sort((a, b) => b[1] - a[1])) {
      const won = draft === t.answer
      parts.push(`<rect x="${PAD}" y="${y - 10}" width="120" height="12" rx="3" fill="${C.chip}"/>`)
      parts.push(`<rect x="${PAD}" y="${y - 10}" width="${Math.max(3, 120 * p)}" height="12" rx="3" fill="${won ? C.orange : C.muted}"/>`)
      parts.push(`<text x="${PAD + 130}" y="${y + 1}" font-family="${SANS}" font-size="14" font-weight="${won ? 700 : 400}" fill="${C.ink}">${Math.round(p * 100)}%  ${esc(clip(draft, 90))}${won ? '  ✓' : ''}</text>`)
      y += 22
    }
    y += 14
  }
  // Every keystroke in order, wrapped across lines, each with a bar for Jev's confidence in it.
  parts.push(`<text x="${PAD}" y="${y + 4}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">KEYSTROKES · BAR = CONFIDENCE</text>`)
  y += 18
  const KEY_H = 26
  const LINE_H = KEY_H + 18
  let x = PAD
  const strip: string[] = []
  for (const step of t.steps) {
    const k = kind(step.pick)
    const text = k === 'backspace' ? '⌫' : withdrawn(step) ? 'SPEAK?✗' : label(step.pick)
    const w = Math.max(KEY_H, textWidth(text, 16) + 14)
    if (x + w > W - PAD) {
      x = PAD
      y += LINE_H
    }
    const fill = withdrawn(step) ? '#fff' : { word: C.orange, letter: '#fff', backspace: '#fff', speak: C.ink }[k]
    const ink = withdrawn(step) ? C.ink : { word: '#fff', letter: C.orange, backspace: C.red, speak: '#fff' }[k]
    const stroke = k === 'letter' ? C.orange : k === 'backspace' ? C.red : withdrawn(step) ? C.ink : fill
    strip.push(`<rect x="${x}" y="${y}" width="${w}" height="${KEY_H}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`)
    strip.push(`<text x="${x + w / 2}" y="${y + 18}" font-family="${k === 'letter' ? MONO : SANS}" font-size="16" font-weight="700" fill="${ink}" text-anchor="middle">${esc(text)}</text>`)
    strip.push(`<rect x="${x}" y="${y + KEY_H + 4}" width="${w}" height="5" rx="2.5" fill="${C.line}"/>`)
    strip.push(`<rect x="${x}" y="${y + KEY_H + 4}" width="${Math.max(2, w * step.confidence)}" height="5" rx="2.5" fill="${C.ink}"/>`)
    x += w + 5
  }
  parts.push(...strip)
  y += LINE_H + 22
  if (!full) return finish(parts, y)

  parts.push(`<text x="${PAD + 36}" y="${y}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">PICKED</text>`)
  parts.push(`<text x="${PAD + 236}" y="${y}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">PASSED OVER</text>`)
  parts.push(`<text x="${W - PAD}" y="${y}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1" text-anchor="end">TEXT SO FAR</text>`)
  y += 14
  for (const i of indices) {
    parts.push(`<line x1="${PAD}" x2="${W - PAD}" y1="${y - 4}" y2="${y - 4}" stroke="${C.line}"/>`)
    parts.push(row(t.steps[i], i, states[i], y))
    y += ROW
    if (i === gapAfter) {
      parts.push(`<text x="${W / 2}" y="${y + 14}" font-family="${SANS}" font-size="13" fill="${C.muted}" text-anchor="middle">… ${t.steps.length - MAX_ROWS} more picks …</text>`)
      y += ROW
    }
  }
  return finish(parts, y + 16)
}

// Adds the footer and wraps the parts in an SVG sized to fit.
function finish(parts: string[], y: number): string {
  parts.push(`<text x="${W - PAD}" y="${y}" font-family="${SANS}" font-size="12" fill="${C.muted}" text-anchor="end">@jevons-talking.bsky.social · github.com/kmosher/jevons-talking</text>`)
  const H = y + PAD - 12
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${C.bg}"/>${parts.join('')}</svg>`
}

export function renderPng(t: Talk, zoom = 2, full = false): { png: Buffer; width: number; height: number } {
  const r = new Resvg(renderSvg(t, { full }), { fitTo: { mode: 'zoom', value: zoom }, font: { loadSystemFonts: true, defaultFontFamily: 'Helvetica Neue' } }).render()
  return { png: r.asPng(), width: r.width, height: r.height }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const [input, out = 'trace.png'] = args.filter((a) => !a.startsWith('--'))
  const { png } = renderPng(JSON.parse(readFileSync(input, 'utf8')), 2, args.includes('--full'))
  writeFileSync(out, png)
  console.log(`wrote ${out} (${(png.length / 1024).toFixed(0)} KB)`)
}
