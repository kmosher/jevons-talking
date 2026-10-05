// Bluesky bot: answers @-mentions and DMs from allowed accounts by running hybrid().
// A post is answered only if it @-mentions the bot or replies directly to one of its posts;
// Bluesky also notifies about other replies anywhere in a thread the bot has posted in.
// A mention in a thread is answered with the thread back to its root as context (the root
// plus the most recent posts, within THREAD_POSTS and THREAD_CHARS); DMs are answered one
// message at a time. Mention replies carry a rendered trace image.
// Polls every 30s and handles one question at a time. On its first run it marks
// everything already there as handled, so it only answers messages sent after that.
//
// Anyone may ask, within PER_USER_DAILY questions each (FRIEND_DAILY for people connected to an
// owner by a follow either way) and GLOBAL_DAILY in total per UTC day;
// past either limit the bot says so once and then stays quiet until the next day.
//
// Env: BLUESKY_HANDLE, BLUESKY_APP_PASSWORD, ALLOWED_HANDLES (comma-separated, or * for anyone), OWNER_HANDLES, TYPESAFE_API_KEY
// Usage: npm run bot
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { AppBskyFeedDefs, AtpAgent, RichText } from '@atproto/api'
import type { BeamTalk } from './beam.ts'
import { hybrid } from './hybrid.ts'
import { renderPng } from './render.ts'
import { containsBlocked, meanConfidence, type Post, save, type Step, type Talk } from './talk.ts'

const HANDLE = process.env.BLUESKY_HANDLE ?? 'jevons-talking.bsky.social'
const PASSWORD = process.env.BLUESKY_APP_PASSWORD
const ALLOWED = (process.env.ALLOWED_HANDLES ?? '*').split(',').map((h) => h.trim())
const OPEN = ALLOWED.includes('*')
// Owners are exempt from the daily limits.
const OWNERS = (process.env.OWNER_HANDLES ?? 'mosheroperandi.bsky.social').split(',').map((h) => h.trim())
const PER_USER_DAILY = 5
// For anyone who follows an owner or whom an owner follows.
const FRIEND_DAILY = 24
const GLOBAL_DAILY = 100
const POLL_MS = 30_000
const POST_LIMIT = 300
const DM_LIMIT = 1000
const STATE_FILE = 'bot-state.json'
const THREAD_POSTS = 16
const THREAD_CHARS = 3000
const IMAGE_LIMIT = 950_000
if (!PASSWORD) throw new Error('BLUESKY_APP_PASSWORD is not set')

// Mention URIs and DM ids already answered (or seen before the bot's first run).
type Usage = { day: string; total: number; perUser: Record<string, number>; warned: string[] }
type State = { handled: string[]; usage?: Usage }
const state: State = existsSync(STATE_FILE) ? JSON.parse(readFileSync(STATE_FILE, 'utf8')) : { handled: [] }
let firstRun = !existsSync(STATE_FILE)
const handled = new Set(state.handled)
const today = () => new Date().toISOString().slice(0, 10)
let usage: Usage = state.usage?.day === today() ? state.usage : { day: today(), total: 0, perUser: {}, warned: [] }
const saveState = () => writeFileSync(STATE_FILE, JSON.stringify({ handled: [...handled].slice(-5000), usage }))
// 'ok' to answer (and counts it), 'warn' to say the limit is reached, 'quiet' once already warned.
async function admit(did: string): Promise<'ok' | 'warn' | 'quiet'> {
  if (ownerDids.has(did)) return 'ok'
  if (usage.day !== today()) usage = { day: today(), total: 0, perUser: {}, warned: [] }
  const limit = (await isFriend(did)) ? FRIEND_DAILY : PER_USER_DAILY
  const result = usage.total >= GLOBAL_DAILY || (usage.perUser[did] ?? 0) >= limit ? (usage.warned.includes(did) ? 'quiet' : 'warn') : 'ok'
  if (result === 'ok') {
    usage.total++
    usage.perUser[did] = (usage.perUser[did] ?? 0) + 1
  } else if (result === 'warn') usage.warned.push(did)
  saveState()
  return result
}
// Whether `did` follows, or is followed by, any owner. Cached for the bot's lifetime.
const friends = new Map<string, boolean>()
async function isFriend(did: string): Promise<boolean> {
  if (!friends.has(did)) {
    let friend = false
    for (const owner of ownerDids) {
      const { data } = await agent.app.bsky.graph.getRelationships({ actor: owner, others: [did] })
      const rel = data.relationships[0] as { following?: string; followedBy?: string } | undefined
      if (rel?.following || rel?.followedBy) friend = true
    }
    friends.set(did, friend)
  }
  return friends.get(did)!
}
const LIMIT_REPLY = 'I’m out of words for today. Try again tomorrow.'
const markHandled = (id: string) => {
  handled.add(id)
  saveState()
}

const agent = new AtpAgent({ service: 'https://bsky.social' })
await agent.login({ identifier: HANDLE, password: PASSWORD })
const chat = agent.withProxy('bsky_chat', 'did:web:api.bsky.chat')
const allowedDids = new Set(await Promise.all(ALLOWED.filter((h) => h !== '*').map(async (handle) => (await agent.resolveHandle({ handle })).data.did)))
const ownerDids = new Set(await Promise.all(OWNERS.map(async (handle) => (await agent.resolveHandle({ handle })).data.did)))
const isAllowed = (did: string) => did !== agent.session?.did && (OPEN || allowedDids.has(did))
const log = (...a: unknown[]) => console.log(new Date().toISOString(), ...a)
log(`logged in as ${HANDLE}; answering ${OPEN ? 'anyone' : ALLOWED.join(', ')}`)

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
  const t = await hybrid(question, { conversation, log: (l) => log(l) })
  log(`A: ${t.answer} (${save(t)})`)
  return t
}

// --- Mentions ------------------------------------------------------------------
type Ref = { uri: string; cid: string }

const stripHandles = (text: string) => text.replace(/@[\w.-]+/g, '').trim()

// Posts a message points at, by quote embed or by a bsky.app link, as context entries: Jev
// otherwise sees only a truncated URL.
type Linking = { embed?: { $type?: string; record?: { uri?: string; record?: { uri?: string } } }; facets?: { features: { $type: string; uri?: string }[] }[] }
const POST_LINK = /^https:\/\/bsky\.app\/profile\/([^/]+)\/post\/([^/?#]+)/
async function linkedPosts(msg: Linking): Promise<Post[]> {
  const uris = new Set<string>()
  const quoted = msg.embed?.record?.uri ?? msg.embed?.record?.record?.uri
  if (quoted) uris.add(quoted)
  for (const f of msg.facets ?? [])
    for (const feature of f.features) {
      const m = feature.$type === 'app.bsky.richtext.facet#link' ? feature.uri?.match(POST_LINK) : null
      if (!m) continue
      try {
        const did = m[1].startsWith('did:') ? m[1] : (await agent.resolveHandle({ handle: m[1] })).data.did
        uris.add(`at://${did}/app.bsky.feed.post/${m[2]}`)
      } catch {
        // An unresolvable handle just means no context from that link.
      }
    }
  if (!uris.size) return []
  const { data } = await agent.getPosts({ uris: [...uris].slice(0, 5) })
  return data.posts.map((p) => {
    // Image descriptions, where the poster wrote them, stand in for the images.
    const images = (p.embed as { images?: { alt?: string }[]; media?: { images?: { alt?: string }[] } } | undefined)
    const alts = [...(images?.images ?? []), ...(images?.media?.images ?? [])].map((i) => i.alt?.trim()).filter(Boolean)
    const text = stripHandles((p.record as { text?: string }).text ?? '')
    return { author: `@${p.author.handle} (linked post)`, text: alts.length ? `${text}\n[image: ${alts.join('; ')}]` : text }
  })
}
// The question with bare links removed; a message that was only a link asks about the linked post.
const withoutLinks = (text: string) => text.replace(/\S*bsky\.app\/profile\/\S+/g, '').trim()

async function post(text: string, root: Ref, parent: Ref, embed?: { $type: string }): Promise<Ref> {
  const rt = new RichText({ text })
  await rt.detectFacets(agent)
  return agent.post({ text: rt.text, facets: rt.facets, reply: { root, parent }, ...(embed ? { embed } : {}) })
}

// The trace as an image embed, re-rendered smaller if it's over Bluesky's size limit.
async function traceImage(t: Talk, full = false) {
  let img = renderPng(t, 2, full)
  for (const zoom of [1.5, 1]) if (img.png.length > IMAGE_LIMIT) img = renderPng(t, zoom, full)
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
    if (firstRun || !isAllowed(n.author.did)) {
      markHandled(n.uri)
      continue
    }
    const record = n.record as { text: string; reply?: { root: Ref; parent: Ref }; facets?: { features: { $type: string; did?: string }[] }[] }
    const me = agent.session!.did
    const mentioned = record.facets?.some((f) => f.features.some((x) => x.$type === 'app.bsky.richtext.facet#mention' && x.did === me))
    const repliesToMe = record.reply?.parent.uri.startsWith(`at://${me}/`)
    if (!mentioned && !repliesToMe) {
      markHandled(n.uri)
      continue
    }
    // An owner can add #full to get the full per-pick table in the trace image.
    const full = ownerDids.has(n.author.did) && /#full\b/i.test(record.text)
    const linked = await linkedPosts(n.record as Linking)
    const asked = withoutLinks(stripHandles(record.text.replace(/#full\b/gi, '')))
    const question = asked || (linked.length ? 'What do you make of the linked post?' : '')
    markHandled(n.uri)
    if (!question) continue
    const parent = { uri: n.uri, cid: n.cid }
    const root = record.reply?.root ?? parent
    const admitted = await admit(n.author.did)
    if (admitted !== 'ok') {
      log(`over limit: @${n.author.handle} (${admitted})`)
      if (admitted === 'warn') await post(LIMIT_REPLY, root, parent)
      continue
    }
    const thread = record.reply ? await threadContext(n.uri) : []
    const context = [...thread, ...linked]
    const t = await answer(question, context.length ? context : undefined)
    const image = containsBlocked(t.answer) ? undefined : await traceImage(t, full)
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
    if (!convo.members.some((m) => isAllowed(m.did))) continue
    if (!firstRun && convo.unreadCount === 0) continue
    const { data: msgs } = await chat.chat.bsky.convo.getMessages({ convoId: convo.id, limit: 20 })
    for (const m of msgs.messages.reverse()) {
      if (m.$type !== 'chat.bsky.convo.defs#messageView') continue
      const msg = m as { id: string; text: string; sender: { did: string } } & Linking
      if (handled.has(msg.id)) continue
      markHandled(msg.id)
      if (firstRun || !isAllowed(msg.sender.did) || !msg.text.trim()) continue
      const send = (text: string) => chat.chat.bsky.convo.sendMessage({ convoId: convo.id, message: { text } })
      const admitted = await admit(msg.sender.did)
      if (admitted !== 'ok') {
        if (admitted === 'warn') await send(LIMIT_REPLY)
        continue
      }
      const linked = await linkedPosts(msg)
      const asked = withoutLinks(msg.text)
      const t = await answer(asked || 'What do you make of the linked post?', linked.length ? linked : undefined)
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
