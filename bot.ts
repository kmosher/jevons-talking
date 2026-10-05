// Bluesky bot: answers @-mentions and DMs from allowed accounts by running talk().
// Polls every 30s and handles one question at a time. On its first run it marks
// everything already there as handled, so it only answers messages sent after that.
//
// Env: BLUESKY_HANDLE, BLUESKY_APP_PASSWORD, ALLOWED_HANDLES (comma-separated), TYPESAFE_API_KEY
// Usage: npm run bot
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { AtpAgent, RichText } from '@atproto/api'
import { containsBlocked, meanConfidence, save, type Talk, talk } from './talk.ts'

const HANDLE = process.env.BLUESKY_HANDLE ?? 'jevons-talking.bsky.social'
const PASSWORD = process.env.BLUESKY_APP_PASSWORD
const ALLOWED = (process.env.ALLOWED_HANDLES ?? 'mosheroperandi.bsky.social').split(',').map((h) => h.trim())
const POLL_MS = 30_000
const POST_LIMIT = 300
const DM_LIMIT = 1000
const STATE_FILE = 'bot-state.json'
if (!PASSWORD) throw new Error('BLUESKY_APP_PASSWORD is not set')

// Mention URIs and DM ids already answered (or seen before the bot's first run).
type State = { handled: string[] }
const state: State = existsSync(STATE_FILE) ? JSON.parse(readFileSync(STATE_FILE, 'utf8')) : { handled: [] }
let firstRun = !existsSync(STATE_FILE)
const handled = new Set(state.handled)
const markHandled = (id: string) => {
  handled.add(id)
  writeFileSync(STATE_FILE, JSON.stringify({ handled: [...handled].slice(-5000) }))
}

const agent = new AtpAgent({ service: 'https://bsky.social' })
await agent.login({ identifier: HANDLE, password: PASSWORD })
const chat = agent.withProxy('bsky_chat', 'did:web:api.bsky.chat')
const allowedDids = new Set(await Promise.all(ALLOWED.map(async (handle) => (await agent.resolveHandle({ handle })).data.did)))
const log = (...a: unknown[]) => console.log(new Date().toISOString(), ...a)
log(`logged in as ${HANDLE}; answering ${ALLOWED.join(', ')}`)

// --- Formatting ----------------------------------------------------------------
const pickLabel = (pick: string) =>
  pick === 'backspace' ? '⌫' : pick === 'SPEAK' ? '🔊' : pick.replace(/^(word|letter): /, '')

function headline(t: Talk): string {
  if (containsBlocked(t.answer)) return '(Jev composed something I won’t post.)'
  const answer = t.answer || '…'
  const tail = `${t.steps.length} picks · mean confidence ${meanConfidence(t).toFixed(2)}${t.finished ? '' : ' · ran out of picks'}`
  return `${answer}\n\n${tail}`
}

// Picks joined into chunks of at most `limit` characters.
function traceChunks(t: Talk, limit: number): string[] {
  const chunks: string[] = []
  let cur = 'Picks:'
  for (const label of t.steps.map((s) => pickLabel(s.pick))) {
    if (cur.length + 1 + label.length > limit) {
      chunks.push(cur)
      cur = label
    } else cur += ` ${label}`
  }
  chunks.push(cur)
  return chunks
}

async function answer(question: string): Promise<Talk> {
  log(`Q: ${question}`)
  const t = await talk(question, { log: (l) => log(l) })
  log(`A: ${t.answer} (${save(t)})`)
  return t
}

// --- Mentions ------------------------------------------------------------------
type Ref = { uri: string; cid: string }

async function post(text: string, root: Ref, parent: Ref): Promise<Ref> {
  const rt = new RichText({ text })
  await rt.detectFacets(agent)
  return agent.post({ text: rt.text, facets: rt.facets, reply: { root, parent } })
}

async function pollMentions() {
  const { data } = await agent.listNotifications({ reasons: ['mention', 'reply'], limit: 50 })
  for (const n of data.notifications.reverse()) {
    if (handled.has(n.uri)) continue
    if (firstRun || !allowedDids.has(n.author.did)) {
      markHandled(n.uri)
      continue
    }
    const record = n.record as { text: string; reply?: { root: Ref } }
    const question = record.text.replace(/@[\w.-]+/g, '').trim()
    markHandled(n.uri)
    if (!question) continue
    const t = await answer(question)
    const parent = { uri: n.uri, cid: n.cid }
    const root = record.reply?.root ?? parent
    let last = await post(headline(t), root, parent)
    for (const chunk of traceChunks(t, POST_LIMIT)) last = await post(chunk, root, last)
  }
  await agent.updateSeenNotifications()
}

// --- DMs -----------------------------------------------------------------------
async function pollDms() {
  const { data } = await chat.chat.bsky.convo.listConvos({ limit: 50 })
  for (const convo of data.convos) {
    if (!convo.members.some((m) => allowedDids.has(m.did))) continue
    if (!firstRun && convo.unreadCount === 0) continue
    const { data: msgs } = await chat.chat.bsky.convo.getMessages({ convoId: convo.id, limit: 20 })
    for (const m of msgs.messages.reverse()) {
      if (m.$type !== 'chat.bsky.convo.defs#messageView') continue
      const msg = m as { id: string; text: string; sender: { did: string } }
      if (handled.has(msg.id)) continue
      markHandled(msg.id)
      if (firstRun || !allowedDids.has(msg.sender.did) || !msg.text.trim()) continue
      const t = await answer(msg.text.trim())
      const send = (text: string) => chat.chat.bsky.convo.sendMessage({ convoId: convo.id, message: { text } })
      await send(headline(t))
      for (const chunk of traceChunks(t, DM_LIMIT)) await send(chunk)
    }
    await chat.chat.bsky.convo.updateRead({ convoId: convo.id })
  }
}

// --- Loop ----------------------------------------------------------------------
for (;;) {
  for (const [name, poll] of [['mentions', pollMentions], ['dms', pollDms]] as const) {
    try {
      await poll()
    } catch (e) {
      log(`${name} poll failed:`, e instanceof Error ? e.message : e)
    }
  }
  if (firstRun) log('first run: marked existing mentions and DMs as handled')
  firstRun = false
  await new Promise((r) => setTimeout(r, POLL_MS))
}
