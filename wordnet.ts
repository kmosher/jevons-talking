// A thesaurus from WordNet's own files (the wordnet-db package): a word's synonyms, plus the
// "similar to" adjectives and the broader terms (hypernyms) of each of its senses.
import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync, renameSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { gunzipSync } from 'node:zlib'
import { createRequire } from 'node:module'
import { join } from 'node:path'

const { path: DICT } = createRequire(import.meta.url)('wordnet-db') as { path: string }
const POS = ['noun', 'verb', 'adj', 'adv'] as const

// A text file of lines sorted by their leading key (up to `sep`), binary-searched on disk like
// look(1): a lookup costs a few small reads and nothing stays in memory.
class SortedFile {
  private fd?: number
  private size = 0
  constructor(private path: () => string, private sep: string) {}
  private open() {
    if (this.fd === undefined) {
      this.fd = openSync(this.path(), 'r')
      this.size = fstatSync(this.fd).size
    }
    return this.fd
  }
  // The first whole line starting at or after byte `pos`.
  private lineFrom(pos: number): string | undefined {
    const fd = this.open()
    let start = pos
    if (pos > 0) {
      const buf = Buffer.alloc(256)
      for (let at = pos - 1; ; at += buf.length) {
        const n = readSync(fd, buf, 0, buf.length, at)
        if (n <= 0) return undefined
        const nl = buf.subarray(0, n).indexOf(10)
        if (nl >= 0) {
          start = at + nl + 1
          break
        }
      }
    }
    if (start >= this.size) return undefined
    const parts: Buffer[] = []
    for (let at = start; at < this.size; ) {
      const buf = Buffer.alloc(8192)
      const n = readSync(fd, buf, 0, buf.length, at)
      if (n <= 0) break
      const nl = buf.subarray(0, n).indexOf(10)
      parts.push(buf.subarray(0, nl >= 0 ? nl : n))
      if (nl >= 0) break
      at += n
    }
    return Buffer.concat(parts).toString('utf8')
  }
  private keyOf = (line: string) => line.slice(0, line.indexOf(this.sep))
  // The line whose key is exactly `key`, if any.
  find(key: string): string | undefined {
    this.open()
    let lo = 0
    let hi = this.size
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2)
      const line = this.lineFrom(mid)
      if (line === undefined || this.keyOf(line) >= key) hi = mid
      else lo = mid + 1
    }
    const line = this.lineFrom(lo)
    return line !== undefined && this.keyOf(line) === key ? line : undefined
  }
}

// WordNet's index files are sorted by lemma (their license header lines start with spaces, which
// sort first). A lemma's line lists the offsets of its senses in the matching data file.
const indexes = Object.fromEntries(POS.map((pos) => [pos, new SortedFile(() => join(DICT, `index.${pos}`), ' ')])) as Record<(typeof POS)[number], SortedFile>
function senses(lemma: string): { pos: (typeof POS)[number]; offsets: number[] }[] {
  return POS.flatMap((pos) => {
    const line = indexes[pos].find(lemma)
    if (!line) return []
    const f = line.trim().split(' ')
    return [{ pos, offsets: f.slice(-Number(f[2])).map(Number) }]
  })
}

function synset(pos: (typeof POS)[number], offset: number): { words: string[]; pointers: { sym: string; offset: number; pos: string }[] } {
  const fd = openSync(join(DICT, `data.${pos}`), 'r')
  const buf = Buffer.alloc(4096)
  const n = readSync(fd, buf, 0, buf.length, offset)
  closeSync(fd)
  const f = buf.subarray(0, n).toString('utf8').split('\n')[0].split(' ')
  const wCnt = parseInt(f[3], 16)
  const words = Array.from({ length: wCnt }, (_, i) => f[4 + i * 2].replace(/\(.*\)$/, ''))
  let i = 4 + wCnt * 2
  const pCnt = Number(f[i++])
  const pointers = Array.from({ length: pCnt }, (_, k) => ({ sym: f[i + k * 4], offset: Number(f[i + k * 4 + 1]), pos: f[i + k * 4 + 2] }))
  return { words, pointers }
}

const POS_OF: Record<string, (typeof POS)[number]> = { n: 'noun', v: 'verb', a: 'adj', s: 'adj', r: 'adv' }
// Plain inflections to look up when the word itself isn't a WordNet lemma.
const lemmas = (w: string) => [w, w.replace(/ies$/, 'y'), w.replace(/es$/, ''), w.replace(/s$/, ''), w.replace(/ed$/, ''), w.replace(/ed$/, 'e'), w.replace(/ing$/, ''), w.replace(/ing$/, 'e')]

// The Moby Thesaurus (public domain, ~30k roots, Project Gutenberg #3202), shipped in data/ as
// one "root,syn,syn,…" line per root, lowercased and byte-sorted. It is unzipped once to a cache
// file and binary-searched on disk like look(1), so nothing stays in memory. JEV_MOBY=off leaves
// it out. Its lists are alphabetical, so mutual synonyms (each lists the other) come first.
const MOBY_GZ = new URL('data/mthesaur.txt.gz', import.meta.url).pathname
function mobyPath(): string {
  const cache = join(process.env.JEV_CACHE ?? tmpdir(), `jevons-talking-mthesaur-${statSync(MOBY_GZ).size}.txt`)
  if (!existsSync(cache)) {
    const part = `${cache}.${process.pid}`
    writeFileSync(part, gunzipSync(readFileSync(MOBY_GZ)))
    renameSync(part, cache)
  }
  return cache
}
const moby = new SortedFile(mobyPath, ',')
const mobyList = (w: string) => moby.find(w)?.slice(w.length + 1).split(',') ?? []
function mobyFor(w: string): string[] {
  if (process.env.JEV_MOBY === 'off') return []
  const syns = mobyList(w)
  const mutual = syns.slice(0, 60).filter((x) => mobyList(x).includes(w))
  return [...mutual, ...syns.filter((x) => !mutual.includes(x))]
}

// Single-word alternatives for `word`, synonyms first; at most `n`.
export function thesaurus(word: string, n = 40): string[] {
  const w = word.toLowerCase().replace(/[^a-z'-]/g, '')
  const entries = lemmas(w).flatMap(senses)
  const near: string[] = []
  const broader: string[] = []
  for (const { pos, offsets } of entries)
    for (const off of offsets) {
      const s = synset(pos, off)
      near.push(...s.words)
      for (const p of s.pointers)
        if (p.sym === '&' || p.sym === '@') (p.sym === '&' ? near : broader).push(...synset(POS_OF[p.pos], p.offset).words)
    }
  const mobySyns = lemmas(w).flatMap(mobyFor)
  // Interleave WordNet's near synonyms with Moby's, then the broader terms.
  const mixed = Array.from({ length: Math.max(near.length, mobySyns.length) }, (_, i) => [near[i], mobySyns[i]]).flat().filter(Boolean) as string[]
  return [...new Set([...mixed, ...broader].map((x) => x.toLowerCase()).filter((x) => /^[a-z'-]+$/.test(x) && x !== w))].slice(0, n)
}

if (import.meta.main) for (const w of process.argv.slice(2)) console.log(w, '→', thesaurus(w).join(', '))
