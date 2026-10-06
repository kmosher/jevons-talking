// Talk to JT the way the bot would, without Bluesky: the full reply pipeline, the trace image
// and its alt text.
//
//   npm run jt -- "is a hot dog a sandwich?"
//   npm run jt -- "what do you see?" --thread thread.json --png out.png --full --log
//   npm run jt                                  # a conversation: each reply joins the thread
//
// --thread is a JSON array of context posts ({ author, role?, text, images?, rejected_drafts? }),
// oldest first; the question is appended as the asker's post. --png writes the trace image
// (default: next to the transcript), --full renders the per-pick table, --log streams Jev's picks.
// With no question, JT answers line by line, and every exchange is added to the thread as the bot
// would see it on a reply: earlier messages as thread posts, JT's replies as "you" with the drafts
// it passed over. --save writes the thread after each reply, to pick it up again with --thread.
import { readFileSync, writeFileSync } from 'node:fs'
import { createInterface } from 'node:readline/promises'
import { parseArgs } from 'node:util'
import { hybrid } from './hybrid.ts'
import { altText, rejectedDrafts, renderPng } from './render.ts'
import { type Post, save } from './talk.ts'

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { thread: { type: 'string' }, png: { type: 'string' }, full: { type: 'boolean' }, log: { type: 'boolean' }, save: { type: 'string' } },
})
const thread: Post[] = values.thread ? JSON.parse(readFileSync(values.thread, 'utf8')) : []

async function ask(question: string, png?: string) {
  const conversation = [...thread, { author: '@asker', role: 'asker' as const, text: question }]
  const t = await hybrid(question, { conversation, log: values.log ? (l) => console.error(l) : undefined })
  const transcript = save(t)
  const out = png ?? transcript.replace(/\.json$/, '.png')
  writeFileSync(out, renderPng(t, 2, values.full).png)
  const alt = altText(t)
  const passed = rejectedDrafts(alt)
  thread.push({ author: '@asker', role: 'thread', text: question }, { author: 'you', role: 'you', text: t.answer, ...(passed.length ? { rejected_drafts: passed } : {}) })
  if (values.save) writeFileSync(values.save, JSON.stringify(thread, null, 2))
  return { t, alt, transcript, png: out }
}

const question = positionals.join(' ').trim()
if (question) {
  const r = await ask(question, values.png)
  console.log(`${r.t.answer}\n\n${r.alt}\n\ntranscript: ${r.transcript}\ntrace: ${r.png}`)
} else {
  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: 'you> ' })
  let open = true
  rl.on('close', () => {
    open = false
  })
  rl.prompt()
  for await (const raw of rl) {
    const line = raw.trim()
    if (line) {
      const r = await ask(line)
      console.log(`jt> ${r.t.answer}\n    (${r.t.path}, ${r.t.calls} calls · trace ${r.png})`)
    }
    if (open) rl.prompt()
  }
}
