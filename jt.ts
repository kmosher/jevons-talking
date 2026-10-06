// Ask JT one question the way the bot would, without Bluesky: the full reply pipeline, the
// trace image and its alt text.
//
//   npm run jt -- "is a hot dog a sandwich?"
//   npm run jt -- "what do you see?" --thread thread.json --png out.png --full --log
//
// --thread is a JSON array of context posts ({ author, role?, text, images?, rejected_drafts? }),
// oldest first; the question is appended as the asker's post. --png writes the trace image
// (default: next to the transcript), --full renders the per-pick table, --log streams Jev's picks.
import { readFileSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { hybrid } from './hybrid.ts'
import { altText, renderPng } from './render.ts'
import { type Post, save } from './talk.ts'

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { thread: { type: 'string' }, png: { type: 'string' }, full: { type: 'boolean' }, log: { type: 'boolean' } },
})
const question = positionals.join(' ').trim()
if (!question) {
  console.error('usage: npm run jt -- "question" [--thread thread.json] [--png out.png] [--full] [--log]')
  process.exit(2)
}
const thread: Post[] = values.thread ? JSON.parse(readFileSync(values.thread, 'utf8')) : []
const conversation = [...thread, { author: '@asker', role: 'asker' as const, text: question }]
const t = await hybrid(question, { conversation, log: values.log ? (l) => console.error(l) : undefined })
const transcript = save(t)
const png = values.png ?? transcript.replace(/\.json$/, '.png')
writeFileSync(png, renderPng(t, 2, values.full).png)
console.log(`${t.answer}\n\n${altText(t)}\n\ntranscript: ${transcript}\ntrace: ${png}`)
