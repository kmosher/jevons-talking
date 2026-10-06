// Draws a PNG inline in the terminal with the kitty graphics protocol (Ghostty, kitty, WezTerm),
// passed through tmux when inside it (needs `set -g allow-passthrough on`). Elsewhere it does
// nothing and returns false.
const CHUNK = 4096

export const canShowImages = () =>
  Boolean(process.stdout.isTTY) &&
  (Boolean(process.env.KITTY_WINDOW_ID || process.env.GHOSTTY_RESOURCES_DIR) || /^(ghostty|kitty|WezTerm)$/.test(process.env.TERM_PROGRAM ?? ''))

export function showImage(png: Buffer, width: number, height: number, maxCols = 100): boolean {
  if (!canShowImages()) return false
  const cols = Math.min(maxCols, (process.stdout.columns ?? 80) - 2)
  // Terminal cells are about twice as tall as wide; both sizes are given so the cursor can be
  // moved past the image ourselves, which tmux, unaware of the image, can't do.
  const rows = Math.ceil((cols * height) / width / 2)
  const tmux = Boolean(process.env.TMUX)
  const wrap = (seq: string) => (tmux ? `\x1bPtmux;${seq.replaceAll('\x1b', '\x1b\x1b')}\x1b\\` : seq)
  const data = png.toString('base64')
  let out = ''
  for (let i = 0; i < data.length; i += CHUNK) {
    const more = i + CHUNK < data.length ? 1 : 0
    const keys = i === 0 ? `a=T,f=100,q=2,C=1,c=${cols},r=${rows},m=${more}` : `m=${more}`
    out += wrap(`\x1b_G${keys};${data.slice(i, i + CHUNK)}\x1b\\`)
  }
  // Make room first (scrolling the pane if needed), then draw into it and step past: under tmux
  // an image drawn at the bottom would not scroll up with the text.
  process.stdout.write(`${'\n'.repeat(rows)}\x1b[${rows}A${out}\x1b[${rows}B\r`)
  return true
}

// node term.ts trace.png: draw a saved PNG, to check the terminal setup.
if (import.meta.main) {
  const { readFileSync } = await import('node:fs')
  const png = readFileSync(process.argv[2])
  if (!showImage(png, png.readUInt32BE(16), png.readUInt32BE(20))) console.error('this terminal can’t draw images (or stdout is not a terminal)')
}
