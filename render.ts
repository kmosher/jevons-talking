// Renders a transcript as a PNG: the question and answer, the drafts a beam search rated, and
// every keystroke Jev made with a bar for how confident it was. With `full`, it adds a table
// with one row per pick: the options passed over and the text as it stood afterwards.
//
// Usage: node --import tsx render.ts <transcript.json> [out.png] [--full]
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { Resvg } from '@resvg/resvg-js'
import type { BeamTalk } from './beam.ts'
import type { HybridTalk } from './hybrid.ts'
import { applyPick, branchText, isSpace, isSpeak, meanConfidence, newBranch, type Step, type Talk } from './talk.ts'

const W = 1080
const PAD = 40
const ROW = 36
const MAX_ROWS = 60
const ALTERNATIVES = 4
const C = { bg: '#f7f4ee', ink: '#1f3a5f', muted: '#8a94a3', orange: '#e8743b', chip: '#dce1e8', line: '#e2ddd2', red: '#c2453d' }
// resvg only tries the first family in a font-family list, so each environment names its own: the
// container's Liberation fonts (see the Dockerfile; DejaVu fills in symbols like ✓), else macOS's.
const LIBERATION = existsSync('/usr/share/fonts/truetype/liberation')
const SANS = LIBERATION ? 'Liberation Sans' : "'Helvetica Neue'"
const SERIF = LIBERATION ? 'Liberation Serif' : 'Georgia'
const MONO = LIBERATION ? 'Liberation Mono' : 'Menlo'

// resvg can't draw colour emoji (one in a line turned the whole headline into boxes), so the
// image shows each as a :name:; the posted text keeps the emoji itself.
const EMOJI_NAMES: Record<string, string> = {
  '🙂': 'smile', '😂': 'joy', '🤔': 'thinking', '😢': 'cry', '😡': 'rage', '😱': 'scream', '👍': '+1', '👎': '-1', '🙏': 'pray',
  '🎉': 'tada', '🔥': 'fire', '✨': 'sparkles', '👀': 'eyes', '🤖': 'robot', '🐢': 'turtle', '🌙': 'moon', '💀': 'skull', '🤷': 'shrug',
}
const deEmoji = (s: string) => s.replace(/\p{Extended_Pictographic}\uFE0F?/gu, (e) => `:${EMOJI_NAMES[e.replace('\uFE0F', '')] ?? 'emoji'}:`)
const esc = (s: string) => deEmoji(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)
// Rough advance width; good enough for sizing chips.
const textWidth = (s: string, size: number) => [...s].reduce((w, ch) => w + (/[A-Z]/.test(ch) ? 0.72 : 0.56), 0) * size

const label = (option: string) => (option === 'backspace' ? '⌫ delete' : isSpeak(option) ? 'SPEAK' : isSpace(option) ? 'SPACE' : option.startsWith('swap: ') ? `↻ ${option.slice(6)}` : option.replace(/^(word|letter|key): /, ''))
// A SPEAK that Jev then said wasn't final.
const withdrawn = (step: Step) => isSpeak(step.pick) && step.final !== undefined && step.final < 0.5
const kind = (option: string) =>
  isSpeak(option) ? 'speak' : option === 'backspace' ? 'backspace' : option.startsWith('letter: ') ? 'letter' : 'word'

// The text (and any letters typed toward the next word) after each step, using talk()'s own editing
// rules; `prefill` is text typed before the first pick (a yes-or-no verdict).
function replay(steps: Step[], prefill: string[] = []): { text: string; prefix: string }[] {
  const b = newBranch(prefill)
  return steps.map((s) => {
    if (!isSpeak(s.pick)) applyPick(b, s.pick)
    return { text: branchText(b), prefix: b.prefix }
  })
}

function chip(x: number, y: number, text: string, opts: { fill: string; ink: string; stroke?: string; size?: number; bold?: boolean; family?: string; opacity?: number }) {
  const size = opts.size ?? 15
  const w = textWidth(deEmoji(text), size) + 18
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
  const v = (t as Partial<HybridTalk>).decisions?.verdict
  const states = replay(t.steps, v ? [v.word, ','] : [])
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
      ...(d.verdict ? [`verdict: ${d.verdict.word} (yes ${Math.round(d.verdict.yes * 100)}%${d.verdict.know !== undefined ? `, knows ${Math.round(d.verdict.know * 100)}%` : ''})`] : []),
      ...(d.contextWords?.some((w) => w.p >= 0.5) ? [`new words: ${d.contextWords.filter((w) => w.p >= 0.5).map((w) => w.word).join(', ')}`] : []),
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
  // Slashdot-style self-moderation: each rubric as five dots, filled to its 0-5 score. A row
  // under the decisions line, or (MOD_LAYOUT footer) along the bottom next to the credit line.
  const scores = (t as Partial<HybridTalk>).scores
  if (scores && MOD_LAYOUT === 'row') {
    parts.push(`<text x="${PAD}" y="${y + 4}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">SELF-MODERATED</text>`)
    modDots(parts, PAD + 132, y, scores, { r: 6.5, step: 17, size: 14, gap: 28 })
    y += 32
  }
  // When there was more than one candidate: each, and how Jev rated it as a final answer.
  const judged = (t as Partial<BeamTalk>).judged
  // A kept rewrite gets its own row under the draft it rewrote, and the ✓ moves to it.
  const rw = (t as Partial<HybridTalk>).rewrite
  const rewritten = rw?.kept === 'after' ? rw : undefined
  if (judged && (Object.keys(judged).length > 1 || rewritten)) {
    parts.push(`<text x="${PAD}" y="${y + 4}" font-family="${SANS}" font-size="12" font-weight="700" fill="${C.muted}" letter-spacing="1">DRAFTS · RATED AS A FINAL ANSWER</text>`)
    y += 26
    for (const [draft, p] of Object.entries(judged).sort((a, b) => b[1] - a[1])) {
      const won = isChosen(t, draft)
      parts.push(`<rect x="${PAD}" y="${y - 10}" width="120" height="12" rx="3" fill="${C.chip}"/>`)
      parts.push(`<rect x="${PAD}" y="${y - 10}" width="${Math.max(3, 120 * p)}" height="12" rx="3" fill="${won ? C.orange : C.muted}"/>`)
      parts.push(`<text x="${PAD + 130}" y="${y + 1}" font-family="${SANS}" font-size="14" font-weight="${won && !rewritten ? 700 : 400}" fill="${C.ink}">${Math.round(p * 100)}%  ${esc(clip(draft, 90))}${won && !rewritten ? '  ✓' : ''}</text>`)
      y += 22
      if (won && rewritten) {
        const q = rewritten.ratings[1]
        const swaps = rewritten.edits.map((e) => `${e.from} → ${e.to || '(deleted)'}`).join(', ')
        parts.push(`<text x="${PAD + 4}" y="${y + 1}" font-family="${SANS}" font-size="14" fill="${C.muted}">↳</text>`)
        parts.push(`<rect x="${PAD + 20}" y="${y - 10}" width="100" height="12" rx="3" fill="${C.chip}"/>`)
        parts.push(`<rect x="${PAD + 20}" y="${y - 10}" width="${Math.max(3, 100 * q)}" height="12" rx="3" fill="${C.orange}"/>`)
        parts.push(`<text x="${PAD + 130}" y="${y + 1}" font-family="${SANS}" font-size="14" font-weight="700" fill="${C.ink}">${Math.round(q * 100)}%  ${esc(clip(rewritten.after, 90))}  ✓<tspan dx="18" font-weight="400" fill="${C.muted}">rewrite: ${esc(swaps)}</tspan></text>`)
        y += 22
      }
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
    const w = Math.max(KEY_H, textWidth(deEmoji(text), 16) + 14)
    if (x + w > W - PAD) {
      x = PAD
      y += LINE_H
    }
    const fill = withdrawn(step) ? '#fff' : { word: C.orange, letter: '#fff', backspace: '#fff', speak: C.ink }[k]
    const ink = withdrawn(step) ? C.ink : { word: '#fff', letter: C.orange, backspace: C.red, speak: '#fff' }[k]
    const stroke = k === 'letter' ? C.orange : k === 'backspace' ? C.red : withdrawn(step) ? C.ink : fill
    strip.push(`<rect x="${x}" y="${y}" width="${w}" height="${KEY_H}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`)
    if (k === 'backspace' && !withdrawn(step)) strip.push(deleteIcon(x + w / 2, y + KEY_H / 2, ink))
    else strip.push(`<text x="${x + w / 2}" y="${y + 18}" font-family="${k === 'letter' ? MONO : SANS}" font-size="16" font-weight="700" fill="${ink}" text-anchor="middle">${esc(text)}</text>`)
    strip.push(`<rect x="${x}" y="${y + KEY_H + 4}" width="${w}" height="5" rx="2.5" fill="${C.line}"/>`)
    strip.push(`<rect x="${x}" y="${y + KEY_H + 4}" width="${Math.max(2, w * step.confidence)}" height="5" rx="2.5" fill="${C.ink}"/>`)
    x += w + 5
  }
  parts.push(...strip)
  y += LINE_H + 22
  if (!full) return finish(parts, y, scores)

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
  return finish(parts, y + 16, scores)
}

// A closer width estimate for short labels than textWidth: narrow and wide letters differ a lot.
const labelWidth = (s: string, size: number) =>
  [...s].reduce((w, ch) => w + (/[iljtfr.' ]/.test(ch) ? 0.3 : /[mwMW]/.test(ch) ? 0.85 : /[A-Z]/.test(ch) ? 0.68 : 0.54), 0) * size
// Five dots per rubric, the last partly filled; returns the x after the last rubric.
function modDots(parts: string[], x: number, y: number, scores: Record<string, number>, o: { r: number; step: number; size: number; gap: number; numbers?: boolean }): number {
  for (const [name, value] of Object.entries(scores)) {
    const label = `${name[0].toUpperCase()}${name.slice(1)}`
    for (let i = 0; i < 5; i++) {
      const cx = x + o.r + i * o.step
      const fill = Math.max(0, Math.min(1, value - i))
      const id = `mod-${name}-${i}`
      parts.push(`<clipPath id="${id}"><circle cx="${cx}" cy="${y}" r="${o.r}"/></clipPath>`)
      parts.push(`<circle cx="${cx}" cy="${y}" r="${o.r}" fill="${C.chip}"/>`)
      if (fill > 0) parts.push(`<rect x="${cx - o.r}" y="${y - o.r}" width="${2 * o.r * fill}" height="${2 * o.r}" fill="${C.orange}" clip-path="url(#${id})"/>`)
      parts.push(`<circle cx="${cx}" cy="${y}" r="${o.r}" fill="none" stroke="${C.ink}" stroke-opacity="0.35" stroke-width="1"/>`)
    }
    x += 4 * o.step + 2 * o.r + 6
    const text = o.numbers ? `${label} ${value.toFixed(1)}` : label
    parts.push(`<text x="${x}" y="${y + o.size * 0.36}" font-family="${SANS}" font-size="${o.size}" fill="${C.ink}">${esc(text)}</text>`)
    x += labelWidth(text, o.size) + o.gap
  }
  return x
}
const MOD_LAYOUT = process.env.JEV_MOD_LAYOUT ?? 'footer'

// Adds the footer and wraps the parts in an SVG sized to fit; with MOD_LAYOUT footer, the
// self-moderation dots sit at its left.
function finish(parts: string[], y: number, scores?: Record<string, number>): string {
  // The bottom strip: the label on its own line, then the dots, with the credit at their right.
  if (scores && MOD_LAYOUT !== 'row') {
    parts.push(`<text x="${PAD}" y="${y}" font-family="${SANS}" font-size="11" font-weight="700" fill="${C.muted}" letter-spacing="1">SELF-MODERATED</text>`)
    modDots(parts, PAD, y + 20, scores, { r: 5, step: 13, size: 12, gap: 22, numbers: MOD_LAYOUT === 'footer-numbers' })
  }
  parts.push(`<text x="${W - PAD}" y="${scores && MOD_LAYOUT !== 'row' ? y + 24 : y}" font-family="${SANS}" font-size="12" fill="${C.muted}" text-anchor="end">@jevons-talking.bsky.social · github.com/kmosher/jevons-talking</text>`)
  const H = y + PAD - 12 + (scores && MOD_LAYOUT !== 'row' ? 22 : 0)
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${C.bg}"/>${parts.join('')}</svg>`
}

export function renderPng(t: Talk, zoom = 2, full = false): { png: Buffer; width: number; height: number } {
  const r = new Resvg(renderSvg(t, { full }), { fitTo: { mode: 'zoom', value: zoom }, font: { loadSystemFonts: true, fontDirs: ['/usr/share/fonts'], defaultFontFamily: LIBERATION ? 'Liberation Sans' : 'Helvetica Neue' } }).render()
  return { png: r.asPng(), width: r.width, height: r.height }
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const [input, out = 'trace.png'] = args.filter((a) => !a.startsWith('--'))
  const { png } = renderPng(JSON.parse(readFileSync(input, 'utf8')), 2, args.includes('--full'))
  writeFileSync(out, png)
  console.log(`wrote ${out} (${(png.length / 1024).toFixed(0)} KB)`)
}

// Whether `draft` is the one that became the answer: the answer may since have been rewritten
// or recased, so compare against the text before the rewrite, ignoring case.
const isChosen = (t: Talk, draft: string) => draft.toLowerCase() === ((t as Partial<HybridTalk>).rewrite?.before ?? t.answer).toLowerCase()

// Slashdot-style moderation, e.g. "Funny 3.2 · Insightful 1.0", strongest first.
const modLine = (s: Record<string, number>) =>
  Object.entries(s)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k[0].toUpperCase()}${k.slice(1)} ${v.toFixed(1)}`)
    .join(' · ')

// The delete key drawn as a shape (a key pointing left with an x in it), since the ⌫ glyph
// looks different in every font that has it.
const deleteIcon = (cx: number, cy: number, color: string) => {
  const w = 16
  const h = 10
  const l = cx - w / 2
  const t = cy - h / 2
  const body = `M ${l} ${cy} L ${l + 5} ${t} H ${l + w} V ${t + h} H ${l + 5} Z`
  const xc = l + 10.5
  const x = `M ${xc - 2} ${cy - 2} L ${xc + 2} ${cy + 2} M ${xc + 2} ${cy - 2} L ${xc - 2} ${cy + 2}`
  return `<path d="${body}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/><path d="${x}" stroke="${color}" stroke-width="1.6" stroke-linecap="round"/>`
}

// --- Alt text ------------------------------------------------------------------
export const pickLabel = (step: Step) =>
  step.pick === 'backspace'
    ? '⌫'
    : isSpeak(step.pick)
      ? step.final !== undefined && step.final < 0.5 ? 'SPEAK(withdrawn)' : 'SPEAK'
      : isSpace(step.pick)
        ? 'SPACE'
        : step.pick.replace(/^(word|letter|key): /, '')

// The trace image's alt text: what was decided before writing, the drafts and how a fresh Jev
// rated them, then every key pressed, cut to fit. The drafts
// line is read back (rejectedDrafts) when JT's own post turns up in a later thread.
const ALT_LIMIT = 1900
const altPct = (p: number) => `${Math.round(p * 100)}%`
export function altText(t: Talk): string {
  const h = t as Partial<HybridTalk>
  const d = h.decisions
  const lines = [`How JT typed "${t.answer}", one key pick at a time.`]
  if (d) {
    const before = [
      `Reply type: ${d.mode}${d.modeConfidence !== undefined ? ` (${altPct(d.modeConfidence)})` : ''}.`,
      d.verdict && `Verdict, asked before typing: ${d.verdict.word} (yes ${altPct(d.verdict.yes)}${d.verdict.know !== undefined ? `, knows the answer ${altPct(d.verdict.know)}` : ''}).`,
      d.contextTotal ? `Kept ${d.contextKept} of ${d.contextTotal} thread posts as relevant.` : '',
      d.images.length ? `Saw ${d.images.length === 1 ? 'an image' : `${d.images.length} images`}: ${d.images.map((i) => `"${i.slice(0, 120)}"`).join('; ')}.` : '',
      d.contextWords?.some((w) => w.p >= 0.5) ? `Words added from the thread: ${d.contextWords.filter((w) => w.p >= 0.5).map((w) => w.word).join(', ')}.` : '',
    ]
    lines.push(before.filter(Boolean).join(' '))
  }
  const judged = Object.entries(h.judged ?? {}).sort((a, b) => b[1] - a[1])
  const chosen = judged.find(([a]) => isChosen(t, a))
  if (h.path === 'beam') {
    const rejected = judged.filter((j) => j !== chosen)
    lines.push(
      `Path: beam search.${chosen ? ` Chosen: "${chosen[0]}" ${altPct(chosen[1])}.` : ''}` +
        (rejected.length ? ` ${REJECTED_LABEL} ${rejected.map(([a, r]) => `"${a}" ${altPct(r)}`).join(', ')}.` : ''),
    )
  } else if (h.path) lines.push(`Path: single draft${judged[0] ? `, rated ${altPct(judged[0][1])} by a fresh Jev` : ''}.`)
  if (h.rewrite?.kept === 'after') lines.push(`Rewritten after: ${h.rewrite.edits.map((e) => `${e.from} → ${e.to || '(deleted)'}`).join(', ')}.`)
  if (h.scores) lines.push(`Self-moderated, 0 to 5: ${modLine(h.scores)}.`)
  lines.push(`${t.steps.length} picks, ${h.calls ?? '?'} Jev calls, mean confidence ${meanConfidence(t).toFixed(2)}.`)
  const alt = lines.join('\n')
  let keys = '\nKeys pressed:'
  for (const s of t.steps) {
    const k = ` ${pickLabel(s)},`
    if (alt.length + keys.length + k.length > ALT_LIMIT - 4) {
      keys += ' …'
      break
    }
    keys += k
  }
  return (alt + keys.replace(/,$/, '.')).slice(0, ALT_LIMIT)
}
// The drafts JT passed over, read back from the alt text of its own trace image.
const REJECTED_LABEL = 'Rejected:'
export const rejectedDrafts = (alt?: string) =>
  [...(alt?.split('\n').find((l) => l.startsWith('Path: beam'))?.split(REJECTED_LABEL)[1]?.matchAll(/"([^"]*)" \d+%/g) ?? [])].map((m) => m[1]).slice(0, 3)

