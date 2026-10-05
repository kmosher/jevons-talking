// Bluesky bot: answers @-mentions and DMs from allowed accounts by running beam().
// A mention in a thread is answered with the thread back to its root as context (the root
// plus the most recent posts, within THREAD_POSTS and THREAD_CHARS); DMs are answered one
// message at a time. Mention replies carry a rendered trace image.
// Polls every 30s and handles one question at a time. On its first run it marks
// everything already there as handled, so it only answers messages sent after that.
//
// Env: BLUESKY_HANDLE, BLUESKY_APP_PASSWORD, ALLOWED_HANDLES (comma-separated), TYPESAFE_API_KEY
// Usage: npm run bot
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { AppBskyFeedDefs, AtpAgent, RichText } from '@atproto/api'
import { type BeamTalk, beam } from './beam.ts'
import { renderPng } from './render.ts'
import { containsBlocked, meanConfidence, type Post, save, type Step, type Talk } from './talk.ts'

const HANDLE = process.env.BLUESKY_HANDLE ?? 'jevons-talking.bsky.social'
const PASSWORD = process.env.BLUESKY_APP_PASSWORD
const ALLOWED = (process.env.ALLOWED_HANDLES ?? 'mosheroperandi.bsky.social').split(',').map((h) => h.trim())
const POLL_MS = 30_000
const POST_LIMIT = 300
const DM_LIMIT = 1000
const STATE_FILE = 'bot-state.json'
const THREAD_POSTS = 16
const THREAD_CHARS = 3000
const IMAGE_LIMIT = 950_000
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
const pickLabel = (step: Step) =>
  step.pick === 'backspace'
    ? '⌫'
    : step.pick === 'SPEAK'
      ? step.final !== undefined && step.final < 0.5 ? '🔊✗' : '🔊'
      : step.pick.replace(/^(word|letter): /, '')

function headline(t: Talk): string {
  if (containsBlocked(t.answer)) return '(Jev composed something I won’t post.)'
  const answer = t.answer || '…'
  const calls = (t as Partial<BeamTalk>).calls
  const tail = `${t.steps.length} picks · ${calls ? `${calls} Jev calls · ` : ''}mean confidence ${meanConfidence(t).toFixed(2)}${t.finished ? '' : ' · ran out of picks'}`
  return `${answer}\n\n${tail}`
}

// Picks joined into chunks of at most `limit` characters.
function traceChunks(t: Talk, limit: number): string[] {
  const chunks: string[] = []
  let cur = 'Picks:'
  for (const label of t.steps.map(pickLabel)) {
    if (cur.length + 1 + label.length > limit) {
      chunks.push(cur)
      cur = label
    } else cur += ` ${label}`
  }
  chunks.push(cur)
  return chunks
}

async function answer(question: string, conversation?: Post[]): Promise<Talk> {
  log(`Q: ${question}${conversation?.length ? ` (+${conversation.length} posts of context)` : ''}`)
  const t = await beam(question, { conversation, log: (l) => log(l) })
  log(`A: ${t.answer} (${save(t)})`)
  return t
}

// --- Mentions ------------------------------------------------------------------
type Ref = { uri: string; cid: string }

const stripHandles = (text: string) => text.replace(/@[\w.-]+/g, '').trim()

async function post(text: string, root: Ref, parent: Ref, embed?: { $type: string }): Promise<Ref> {
  const rt = new RichText({ text })
  await rt.detectFacets(agent)
  return agent.post({ text: rt.text, facets: rt.facets, reply: { root, parent }, ...(embed ? { embed } : {}) })
}

// The trace as an image embed, re-rendered smaller if it's over Bluesky's size limit.
async function traceImage(t: Talk) {
  let img = renderPng(t, 2)
  for (const zoom of [1.5, 1]) if (img.png.length > IMAGE_LIMIT) img = renderPng(t, zoom)
  if (img.png.length > IMAGE_LIMIT) return undefined
  const { data } = await agent.uploadBlob(img.png, { encoding: 'image/png' })
  const alt = traceChunks(t, 100_000)[0].slice(0, 1900)
  return {
    $type: 'app.bsky.embed.images',
    images: [{ image: data.blob, alt, aspectRatio: { width: img.width, height: img.height } }],
  }
}

// The posts above `uri`, root first: the root plus the most recent posts, within the caps.
async function threadContext(uri: string): Promise<Post[]> {
  const { data } = await agent.getPostThread({ uri, depth: 0, parentHeight: THREAD_POSTS })
  const chain: AppBskyFeedDefs.PostView[] = []
  let node: unknown = AppBskyFeedDefs.isThreadViewPost(data.thread) ? data.thread.parent : undefined
  while (AppBskyFeedDefs.isThreadViewPost(node)) {
    const view = node as AppBskyFeedDefs.ThreadViewPost
    chain.unshift(view.post)
    node = view.parent
  }
  // Past the fetch height, fetch the root on its own.
  const top = chain[0]?.record as { reply?: { root: Ref } } | undefined
  const rootUri = top?.reply?.root.uri
  let root: AppBskyFeedDefs.PostView | undefined
  if (rootUri) root = (await agent.getPosts({ uris: [rootUri] })).data.posts[0]

  const toPost = (p: AppBskyFeedDefs.PostView): Post => ({
    author: p.author.did === agent.session?.did ? 'you' : `@${p.author.handle}`,
    text: stripHandles((p.record as { text?: string }).text ?? ''),
  })
  // The root, from the chain if it reached it, else fetched separately.
  const rootPost = root ? toPost(root) : chain.length ? toPost(chain[0]) : undefined
  const rest = (root ? chain : chain.slice(1)).map(toPost)
  // Most recent posts that fit the caps, leaving one slot for the root.
  const kept: Post[] = []
  let chars = rootPost?.text.length ?? 0
  for (let i = rest.length - 1; i >= 0; i--) {
    if (kept.length >= THREAD_POSTS - 1 || chars + rest[i].text.length > THREAD_CHARS) break
    kept.unshift(rest[i])
    chars += rest[i].text.length
  }
  const gap = Boolean(root) || kept.length < rest.length
  return [...(rootPost ? [rootPost] : []), ...(gap ? [{ author: '…', text: '(earlier posts omitted)' }] : []), ...kept]
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
    const question = stripHandles(record.text)
    markHandled(n.uri)
    if (!question) continue
    const context = record.reply ? await threadContext(n.uri) : undefined
    const t = await answer(question, context)
    const parent = { uri: n.uri, cid: n.cid }
    const root = record.reply?.root ?? parent
    const image = containsBlocked(t.answer) ? undefined : await traceImage(t)
    const last = await post(headline(t), root, parent, image)
    if (!image) {
      let prev = last
      for (const chunk of traceChunks(t, POST_LIMIT)) prev = await post(chunk, root, prev)
    }
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
