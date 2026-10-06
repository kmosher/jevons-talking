// A thesaurus from WordNet's own files (the wordnet-db package): a word's synonyms, plus the
// "similar to" adjectives and the broader terms (hypernyms) of each of its senses.
import { closeSync, openSync, readFileSync, readSync } from 'node:fs'
import { gunzipSync } from 'node:zlib'
import { createRequire } from 'node:module'
import { join } from 'node:path'

const { path: DICT } = createRequire(import.meta.url)('wordnet-db') as { path: string }
const POS = ['noun', 'verb', 'adj', 'adv'] as const
let index: Map<string, { pos: (typeof POS)[number]; offsets: number[] }[]> | undefined

function loadIndex() {
  index = new Map()
  for (const pos of POS)
    for (const line of readFileSync(join(DICT, `index.${pos}`), 'utf8').split('\n')) {
      if (!line || line.startsWith(' ')) continue
      const f = line.split(' ')
      const senses = Number(f[2])
      const offsets = f.slice(-senses - 1, -1).map(Number)
      const e = index.get(f[0]) ?? index.set(f[0], []).get(f[0])!
      e.push({ pos, offsets })
    }
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

// The Moby Thesaurus (public domain, ~30k roots, Project Gutenberg #3202), shipped gzipped in
// data/; JEV_MOBY=off leaves it out, or a path points at another copy. Its lists are long and
// alphabetical, so mutual synonyms (each lists the other) come first.
// Entries stay as raw lines (split on lookup): splitting all 2.5M synonyms up front took ~320MB.
let moby: Map<string, string> | undefined
const mobyList = (w: string) => moby!.get(w)?.split(',') ?? []
function mobyFor(w: string): string[] {
  if (process.env.JEV_MOBY === 'off') return []
  if (!moby) {
    moby = new Map()
    const src = process.env.JEV_MOBY ?? new URL('data/mthesaur.txt.gz', import.meta.url).pathname
    const raw = readFileSync(src)
    for (const line of (src.endsWith('.gz') ? gunzipSync(raw) : raw).toString('utf8').toLowerCase().split(/\r?\n/)) {
      const comma = line.indexOf(',')
      if (comma > 0) moby.set(line.slice(0, comma), line.slice(comma + 1))
    }
  }
  const syns = mobyList(w)
  const mutual = syns.filter((x) => moby!.get(x)?.split(',').includes(w))
  return [...mutual, ...syns.filter((x) => !mutual.includes(x))]
}

// Single-word alternatives for `word`, synonyms first; at most `n`.
export function thesaurus(word: string, n = 40): string[] {
  if (!index) loadIndex()
  const w = word.toLowerCase().replace(/[^a-z'-]/g, '')
  const entries = lemmas(w).flatMap((l) => index!.get(l) ?? [])
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
