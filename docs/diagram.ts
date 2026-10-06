// The "how jt talks" diagram (README header and JT's pinned post): node docs/diagram.ts [out.png]
import { writeFileSync } from 'node:fs'
import { Resvg } from '@resvg/resvg-js'

const W = 1200
const NAVY = '#1f3a5f', ORANGE = '#e8743b', BG = '#f7f4ee', INK = '#1f3a5f', MUTED = '#7d8796'
const SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
const SERIF = "Georgia, 'Times New Roman', serif"
const MONO = 'Menlo, monospace'
const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const parts: string[] = []

// A keycap-style box: skirt, face, a small badge saying who decides, a title and detail lines.
function key(x: number, y: number, w: number, h: number, badge: 'JEV' | 'CODE' | 'MODEL', title: string, lines: string[]) {
  const accent = badge === 'JEV' ? ORANGE : badge === 'CODE' ? NAVY : '#8a6fb0'
  parts.push(`<rect x="${x}" y="${y + 6}" width="${w}" height="${h}" rx="18" fill="#bfc4cb"/>`)
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="#fbfbfa" stroke="${accent}" stroke-width="3"${badge === 'MODEL' ? ' stroke-dasharray="10 6"' : ''}/>`)
  parts.push(`<rect x="${x + 16}" y="${y + 16}" width="${badge.length * 11 + 18}" height="24" rx="6" fill="${accent}"/>`)
  parts.push(`<text x="${x + 25}" y="${y + 33}" font-family="${MONO}" font-size="14" font-weight="700" fill="#fff">${badge}</text>`)
  parts.push(`<text x="${x + badge.length * 11 + 46}" y="${y + 35}" font-family="${SANS}" font-size="21" font-weight="700" fill="${INK}">${esc(title)}</text>`)
  lines.forEach((l, i) => {
    parts.push(`<text x="${x + 18}" y="${y + 66 + i * 23}" font-family="${SANS}" font-size="16" fill="${INK}">${esc(l)}</text>`)
  })
}
function arrow(x1: number, y1: number, x2: number, y2: number, label = '') {
  parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2 - 10}" stroke="${NAVY}" stroke-width="3"/>`)
  parts.push(`<path d="M ${x2 - 9} ${y2 - 14} L ${x2} ${y2} L ${x2 + 9} ${y2 - 14} Z" fill="${NAVY}"/>`)
  if (label) parts.push(`<text x="${(x1 + x2) / 2 + 12}" y="${(y1 + y2) / 2 + 5}" font-family="${SANS}" font-size="15" font-style="italic" fill="${MUTED}">${esc(label)}</text>`)
}
function chip(x: number, y: number, text: string, fill: string, ink = '#fff', stroke = fill) {
  const w = text.length * 10 + 18
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="28" rx="7" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`)
  parts.push(`<text x="${x + w / 2}" y="${y + 19}" font-family="${SANS}" font-size="15" font-weight="700" fill="${ink}" text-anchor="middle">${esc(text)}</text>`)
  return w
}

// Title
parts.push(`<text x="60" y="78" font-family="${SERIF}" font-size="46" font-weight="700" fill="${INK}">how jt talks</text>`)
parts.push(`<text x="60" y="112" font-family="${SANS}" font-size="19" fill="${MUTED}">Jev can't write. It can only pick. So it gets a keyboard, and picks one key at a time.</text>`)
// Legend
let lx = 60
lx += chip(lx, 136, 'JEV', ORANGE) + 8
parts.push(`<text x="${lx}" y="155" font-family="${SANS}" font-size="15" fill="${INK}">Jev decides</text>`)
lx += 110
lx += chip(lx, 136, 'CODE', NAVY) + 8
parts.push(`<text x="${lx}" y="155" font-family="${SANS}" font-size="15" fill="${INK}">plain code</text>`)
lx += 100
lx += chip(lx, 136, 'MODEL', '#8a6fb0') + 8
parts.push(`<text x="${lx}" y="155" font-family="${SANS}" font-size="15" fill="${INK}">a small image model: describes pictures, never writes JT's words</text>`)

const X = 60, BW = 1080, HALF = 525
let y = 196
key(X, y, BW, 74, 'CODE', 'someone talks to jt', ['an @mention, a reply to one of jt\'s posts, or a DM. jt likes the post to show it was seen'])
arrow(600, y + 80, 600, y + 110); y += 110
key(X, y, HALF, 132, 'CODE', 'gather context', ['the thread back to its root, linked posts', "jt's own replies as \"you\", with the", 'drafts it passed over'])
key(X + 555, y, HALF, 132, 'MODEL', 'look at images', ['Florence-2 captions each picture and', 'reads any text in it, next to alt text', '(it never sees the conversation)'])
arrow(600, y + 138, 600, y + 168); y += 168
key(X, y, BW, 98, 'JEV', 'sort the context', ['yes/no per post: "relevant to the latest message?" and "do my old drafts help?"', 'words jt has never seen (names, coinages) join the keyboard if jev might use them'])
arrow(600, y + 104, 600, y + 134); y += 134
key(X, y, BW, 98, 'JEV', 'what kind of reply?', ['answer · yes/no · comeback · react · scene · acknowledge, which sets the last line of jt\'s instructions', 'a yes/no question gets jev\'s verdict first ("yes," / "no," / "maybe,"), then jt has to justify it'])
arrow(600, y + 104, 600, y + 134); y += 134

// The keyboard: the heart of it
key(X, y, BW, 262, 'JEV', 'write it, one key at a time', ['each step, jev picks exactly one key from a menu of ~220:'])
let cx = X + 18, cy = y + 82
for (const t of ['→ "i see the surface"', '→ "i see the light"', '→ "i see a"', '…150 predicted words']) cx += chip(cx, cy, t, ORANGE, '#fff', ORANGE) + 6
cx = X + 18; cy += 36
for (const t of ['a', 'b', 'c', '…', 'z', '0-9']) cx += chip(cx, cy, t, '#fff', ORANGE, ORANGE) + 6
cx += 10
for (const t of ['.', ',', '?', '!', '—', '…', '-', '(', ')']) cx += chip(cx, cy, t, '#fff', NAVY, NAVY) + 6
cx = X + 18; cy += 36
cx += chip(cx, cy, 'SPACE', '#fff', NAVY, NAVY) + 6
cx += chip(cx, cy, '⌫ delete', '#fff', '#c2453d', '#c2453d') + 6
cx += chip(cx, cy, 'SPEAK', NAVY) + 6
parts.push(`<text x="${X + 18}" y="${y + 214}" font-family="${SANS}" font-size="15" fill="${MUTED}">each word shows the sentence it would make. with the same pick, jev says how the last word is written: as is, Capitalized or ALL CAPS.</text>`)
parts.push(`<text x="${X + 18}" y="${y + 238}" font-family="${SANS}" font-size="15" fill="${MUTED}">predictions = plain word-pair counts from WordNet's dictionary definitions + contractions. no language model.</text>`)
arrow(600, y + 268, 600, y + 298); y += 298

key(X, y, BW, 106, 'JEV', 'judge the draft (a fresh jev that only sees the question)', ['"is this a good reply?" × (1 − ½ × "is it generic?")', 'repeating the question or jt\'s own earlier replies is marked down · a bare "no" needs 60%'])
const by = y + 112
arrow(330, by, 330, by + 40, 'good enough')
arrow(870, by, 870, by + 40, 'weak, or a short quip')
y = by + 40
key(X, y, HALF, 132, 'CODE', 'keep it', ['most answers end here,', 'in 5–30 picks'])
key(X + 555, y, HALF, 132, 'JEV', 'beam search', ['3 drafts at once, forking on any', 'key jev gives ≥5%; a fresh jev', 'picks the best finished draft'])
arrow(330, y + 138, 560, y + 178)
arrow(870, y + 138, 640, y + 178)
y += 178
key(X, y, BW, 106, 'JEV', 'one last look: rewrite', ['"which words would you replace?" then a pick from the word\'s other forms (is → am/are/was)', 'and a thesaurus (WordNet + Moby). a fresh jev keeps whichever version reads better'])
arrow(600, y + 112, 600, y + 142); y += 142
key(X, y, BW, 80, 'CODE', 'post the reply', ['with a picture of every key jev pressed and how sure it was, plus the drafts it passed over'])
y += 104
parts.push(`<text x="${W - 60}" y="${y}" font-family="${SANS}" font-size="15" fill="${MUTED}" text-anchor="end">@jevons-talking.bsky.social · github.com/kmosher/jevons-talking</text>`)
const H = y + 40
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="${BG}"/>${parts.join('')}</svg>`
const png = new Resvg(svg, { fitTo: { mode: 'zoom', value: 2 }, font: { loadSystemFonts: true, defaultFontFamily: 'Helvetica Neue' } }).render()
const out = process.argv[2] ?? new URL('how-jt-talks.png', import.meta.url).pathname
writeFileSync(out, png.asPng())
console.log('wrote', out, png.width, 'x', png.height, Math.round(png.asPng().length / 1024), 'KB')
