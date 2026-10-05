// Renders a talk() transcript as a PNG: one row per pick showing what Jev chose, how
// confident it was, the options it passed over, and the text as it stood afterwards.
//
// Usage: node --import tsx render.ts <transcript.json> [out.png]
import { readFileSync, writeFileSync } from 'node:fs'
import { Resvg } from '@resvg/resvg-js'
import { meanConfidence, type Step, type Talk } from './talk.ts'

const W = 1080
const PAD = 40
const ROW = 36
const MAX_ROWS = 60
const ALTERNATIVES = 4
const C = { bg: '#f7f4ee', ink: '#1f3a5f', muted: '#8a94a3', orange: '#e8743b', chip: '#e3e6ea', line: '#e2ddd2', red: '#c2453d' }
const SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const SERIF = "Georgia, 'Times New Roman', serif"
const MONO = 'Menlo, Monaco, monospace'

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)
// Rough advance width; good enough for sizing chips.
const textWidth = (s: string, size: number) => [...s].reduce((w, ch) => w + (/[A-Z]/.test(ch) ? 0.72 : 0.56), 0) * size

const label = (option: string) => (option === 'backspace' ? '⌫ delete' : option.replace(/^(word|letter): /, ''))
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
      `<text x="${x + 9}" y="${y + 18}" font-family="${opts.family ?? SANS}" font-size="${size}" font-weight="${opts.bold ? 700 : 400}" fill="${opts.ink}">${esc(text)}</text></g>`,
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
  const pick = chip(PAD + 36, y, clip(label(step.pick), 16), style)
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
    const c = chip(x, y, `${clip(label(option), 12)} ${Math.round(p * 100)}%`, { fill: C.chip, ink: C.ink, size: 13, opacity: 0.2 + 0.8 * Math.sqrt(p) })
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

export function renderSvg(t: Talk): string {
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
  const meta = `${t.steps.length} picks · mean confidence ${meanConfidence(t).toFixed(2)}${t.finished ? '' : ' · ran out of picks'}`
  parts.push(`<text x="${PAD}" y="${top + 88}" font-family="${SANS}" font-size="14" fill="${C.muted}">${esc(meta)}</text>`)

  // Every pick in order, as the text trace shows it, wrapped across lines.
  let y = top + 112
  let x = PAD
  const strip: string[] = []
  for (const step of t.steps) {
    const k = kind(step.pick)
    const text = k === 'backspace' ? '⌫' : label(step.pick)
    const w = textWidth(text, 14) + 12
    if (x + w > W - PAD) {
      x = PAD
      y += 26
    }
    const fill = { word: C.orange, letter: '#fff', backspace: '#fff', speak: C.ink }[k]
    const ink = { word: '#fff', letter: C.orange, backspace: C.red, speak: '#fff' }[k]
    const stroke = k === 'letter' ? C.orange : k === 'backspace' ? C.red : fill
    strip.push(`<rect x="${x}" y="${y}" width="${w}" height="21" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`)
    strip.push(`<text x="${x + 6}" y="${y + 15}" font-family="${k === 'letter' ? MONO : SANS}" font-size="14" font-weight="700" fill="${ink}">${esc(text)}</text>`)
    x += w + 4
  }
  parts.push(...strip)
  y += 21 + 40
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
  y += 16
  parts.push(`<text x="${W - PAD}" y="${y}" font-family="${SANS}" font-size="12" fill="${C.muted}" text-anchor="end">@jevons-talking.bsky.social · github.com/kmosher/jevons-talking</text>`)
  const H = y + PAD - 12
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${C.bg}"/>${parts.join('')}</svg>`
}

export function renderPng(t: Talk): { png: Buffer; width: number; height: number } {
  const r = new Resvg(renderSvg(t), { fitTo: { mode: 'zoom', value: 2 }, font: { loadSystemFonts: true, defaultFontFamily: 'Helvetica Neue' } }).render()
  return { png: r.asPng(), width: r.width, height: r.height }
}

if (import.meta.main) {
  const [input, out = 'trace.png'] = process.argv.slice(2)
  const { png } = renderPng(JSON.parse(readFileSync(input, 'utf8')))
  writeFileSync(out, png)
  console.log(`wrote ${out} (${(png.length / 1024).toFixed(0)} KB)`)
}
