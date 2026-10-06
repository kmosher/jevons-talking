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
import { hostname } from 'node:os'
import { AppBskyFeedDefs, AtpAgent, RichText } from '@atproto/api'
import type { BeamTalk } from './beam.ts'
import { describeImage, loadCaptioner } from './caption.ts'
import { hybrid } from './hybrid.ts'
import { renderPng } from './render.ts'
import { containsBlocked, isSpeak, meanConfidence, type Post, save, type Step, type Talk } from './talk.ts'

const HANDLE = process.env.BLUESKY_HANDLE ?? 'jevons-talking.bsky.social'
const PASSWORD = process.env.BLUESKY_APP_PASSWORD
const ALLOWED = (process.env.ALLOWED_HANDLES ?? '*').split(',').map((h) => h.trim())
const OPEN = ALLOWED.includes('*')
// Owners are exempt from the daily limits.
const OWNERS = (process.env.OWNER_HANDLES ?? 'mosheroperandi.bsky.social').split(',').map((h) => h.trim())
const PER_USER_DAILY = 20
// For anyone who follows an owner or whom an owner follows.
const FRIEND_DAILY = 100
const GLOBAL_DAILY = 500
const POLL_MS = 5_000
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
await loadCaptioner().then(
  () => log('image captioner loaded'),
  (e) => log('image captioner failed to load; will retry on the next image:', e instanceof Error ? e.message : e),
)

// --- Formatting ----------------------------------------------------------------
const pickLabel = (step: Step) =>
  step.pick === 'backspace'
    ? '⌫'
    : isSpeak(step.pick)
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

// Mentions of the bot read as "@you", so Jev can tell when it's addressed without being told its
// name; other handles stay, so it can tell who is talking to whom.
const addressed = (text: string) => text.replace(new RegExp(`@${HANDLE.replace(/\./g, '\\.')}`, 'gi'), '@you').trim()
// The bot's own replies, without the stats line under the answer.
const ownText = (text: string) => text.split('\n\n')[0].trim()

// --- Images --------------------------------------------------------------------
// Every image gets a description from Florence-2 (caption.ts), plus the poster's alt text when
// there is some.
const captions = new Map<string, string | null>()
async function describe(url: string): Promise<string | null> {
  if (captions.has(url)) return captions.get(url)!
  let caption: string | null = null
  try {
    caption = await describeImage(url)
  } catch (e) {
    log('image description failed:', e instanceof Error ? e.message : e)
  }
  if (caption) log(`image described: ${caption}`)
  captions.set(url, caption)
  return caption
}
type ImageView = { fullsize?: string; alt?: string }
async function imageNotes(p: AppBskyFeedDefs.PostView): Promise<string> {
  if (p.author.did === agent.session?.did) return ''
  const embed = p.embed as { images?: ImageView[]; media?: { images?: ImageView[] } } | undefined
  const images = [...(embed?.images ?? []), ...(embed?.media?.images ?? [])].slice(0, 4)
  const notes = await Promise.all(
    images.map(async (i) => {
      const caption = i.fullsize ? await describe(i.fullsize) : null
      const alt = i.alt?.trim()
      return [caption, alt && `alt text: ${alt}`].filter(Boolean).join(' / ') || null
    }),
  )
  return notes.filter(Boolean).map((n) => `\n[image: ${n}]`).join('')
}
// A post as a context entry: the bot's own posts as "you", without stats or trace images.
async function asContext(p: AppBskyFeedDefs.PostView, suffix = ''): Promise<Post> {
  const text = (p.record as { text?: string }).text ?? ''
  if (p.author.did === agent.session?.did) return { author: `you${suffix}`, text: ownText(text) }
  return { author: `@${p.author.handle}${suffix}`, text: addressed(text) + (await imageNotes(p)) }
}

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
  return Promise.all(data.posts.map((p) => asContext(p, ' (linked post)')))
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

  const toPost = (p: AppBskyFeedDefs.PostView) => asContext(p)
  // The root, from the chain if it reached it, else fetched separately.
  const rootPost = root ? await toPost(root) : chain.length ? await toPost(chain[0]) : undefined
  const rest = await Promise.all((root ? chain : chain.slice(1)).map(toPost))
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

// Bluesky occasionally creates no notification for a reply, so every SWEEP_EVERY polls (~2.5 min) the bot
// also reads the replies under its own posts from the last day, and treats any it hasn't handled
// like a notification.
const SWEEP_EVERY = 30
let polls = 0
type Incoming = { uri: string; cid: string; author: { did: string; handle: string }; record: unknown; indexedAt: string }
async function missedReplies(): Promise<Incoming[]> {
  if (polls++ % SWEEP_EVERY) return []
  const me = agent.session!.did
  const { data } = await agent.getAuthorFeed({ actor: me, limit: 20, filter: 'posts_with_replies' })
  const out: Incoming[] = []
  for (const item of data.feed) {
    if (item.post.author.did !== me || Date.parse(item.post.indexedAt) < Date.now() - 86_400_000) continue
    const t: unknown = (await agent.getPostThread({ uri: item.post.uri, depth: 1, parentHeight: 0 })).data.thread
    if (!AppBskyFeedDefs.isThreadViewPost(t)) continue
    for (const r of (t as AppBskyFeedDefs.ThreadViewPost).replies ?? []) {
      if (!AppBskyFeedDefs.isThreadViewPost(r)) continue
      const p = (r as AppBskyFeedDefs.ThreadViewPost).post
      if (p.author.did !== me && !handled.has(p.uri)) out.push({ uri: p.uri, cid: p.cid, author: p.author, record: p.record, indexedAt: p.indexedAt })
    }
  }
  if (out.length) log(`sweep found ${out.length} reply(s) with no notification`)
  return out
}

const failures = new Map<string, number>()
const inFlight = new Set<string>()
const MAX_TRIES = 3
const ERROR_REPLY = 'Something broke while I was typing, sorry. Try me again later?'
const CREDITS_REPLY = 'I’m out of Jev credits, so I can’t pick any words right now. Try again later.'
// Jev refusing for want of credits (402, or a quota/credit message) won't fix itself on retry.
const outOfCredits = (e: unknown) => {
  const status = (e as { status?: number }).status
  return status === 402 || ((status === 429 || status === 403) && /credit|quota|balance|billing|payment/i.test(String((e as Error).message)))
}

async function replyTo(n: Incoming, record: { reply?: { root: Ref; parent: Ref } }, question: string, linked: Post[], full: boolean, root: Ref, parent: Ref) {
  try {
    const thread = record.reply ? await threadContext(n.uri) : []
    // The question itself goes last, labelled with who asked it and with its images described.
    const [view] = (await agent.getPosts({ uris: [n.uri] })).data.posts
    const asker = view ? await asContext(view) : { author: `@${n.author.handle}`, text: question }
    const t = await answer(question, [...thread, ...linked, asker])
    const image = containsBlocked(t.answer) ? undefined : await traceImage(t, full)
    const last = await post(headline(t), root, parent, image)
    if (!image) {
      let prev = last
      for (const chunk of traceChunks(t, POST_LIMIT)) prev = await post(chunk, root, prev)
    }
    failures.delete(n.uri)
  } catch (e) {
    const tries = (failures.get(n.uri) ?? 0) + 1
    failures.set(n.uri, tries)
    const credits = outOfCredits(e)
    log(`reply failed (try ${tries} of ${MAX_TRIES}${credits ? ', out of credits' : ''}) for @${n.author.handle}:`, e instanceof Error ? e.message : e)
    // A failed reply (a Jev 503, a Bluesky hiccup) is retried on later polls; after the last
    // try, or when Jev is out of credits, JT says so instead of going quiet.
    if (tries < MAX_TRIES && !credits) {
      handled.delete(n.uri)
      saveState()
    } else await post(credits ? CREDITS_REPLY : ERROR_REPLY, root, parent).catch(() => {})
  }
}

async function pollMentions() {
  const { data } = await agent.listNotifications({ reasons: ['mention', 'reply'], limit: 50 })
  const seen = new Set(data.notifications.map((n) => n.uri))
  const incoming: Incoming[] = [...data.notifications, ...(await missedReplies()).filter((r) => !seen.has(r.uri))].sort((x, y) =>
    x.indexedAt.localeCompare(y.indexedAt),
  )
  for (const n of incoming) {
    if (handled.has(n.uri) || inFlight.has(n.uri)) continue
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
    const asked = withoutLinks(addressed(record.text.replace(/#full\b/gi, '')).replace(/^(@you\b[\s,:]*)+/i, '')).trim()
    const question = asked || (linked.length ? 'What do you make of the linked post?' : '')
    markHandled(n.uri)
    if (!question) continue
    const parent = { uri: n.uri, cid: n.cid }
    const root = record.reply?.root ?? parent
    // A retry was already admitted and liked; only a first attempt counts against the limits.
    if (!failures.has(n.uri)) {
      const admitted = await admit(n.author.did)
      if (admitted !== 'ok') {
        log(`over limit: @${n.author.handle} (${admitted})`)
        if (admitted === 'warn') await post(LIMIT_REPLY, root, parent)
        continue
      }
      // A like on the question says JT has seen it and is typing.
      await agent.like(n.uri, n.cid).catch((e) => log('like failed:', e instanceof Error ? e.message : e))
    }
    // Replies run concurrently: each is almost all waiting on Jev, so there's nothing to gain
    // by queueing one behind another's beam search.
    inFlight.add(n.uri)
    void replyTo(n, record, question, linked, full, root, parent).finally(() => inFlight.delete(n.uri))
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
      const asked = withoutLinks(msg.text) || 'What do you make of the linked post?'
      const handle = convo.members.find((m) => m.did === msg.sender.did)?.handle ?? 'someone'
      const t = await answer(asked, [...linked, { author: `@${handle}`, text: msg.text }])
      await send(headline(t))
      for (const chunk of traceChunks(t, DM_LIMIT)) await send(chunk)
    }
    await chat.chat.bsky.convo.updateRead({ convoId: convo.id })
  }
}

// --- Lease -----------------------------------------------------------------------
// Only one running copy may answer, or every question gets two replies. The copy that holds a
// lease record in JT's own repo answers; others wait and take over once it lapses. The PDS's
// swapRecord makes taking and renewing it a compare-and-swap, so two copies can't both win.
// --force (or JEV_FORCE_LEASE=1) takes it regardless, for when the old copy is known dead.
const LEASE = { repo: agent.session!.did, collection: 'dev.kmosher.jt.lease', rkey: 'self' }
const LEASE_TTL_MS = 5 * 60_000
const LEASE_RENEW_MS = 2 * 60_000
const STANDBY_CHECK_MS = 30_000
const INSTANCE = process.env.JEV_INSTANCE ?? `${hostname()}:${process.pid}`
let force = process.argv.includes('--force') || process.env.JEV_FORCE_LEASE === '1'
let leaseUntil = 0
let lastRenew = 0
type Lease = { instance: string; expiresAt: string }

async function holdLease(): Promise<boolean> {
  const now = Date.now()
  // Between renewals, a cheap read each poll confirms nobody has --forced the lease away.
  if (now < leaseUntil && now - lastRenew < LEASE_RENEW_MS) {
    try {
      const { data } = await agent.com.atproto.repo.getRecord(LEASE)
      const holder = (data.value as unknown as Lease).instance
      if (holder === INSTANCE) return true
      log(`lease lost to ${holder}; standing by`)
      leaseUntil = 0
      lastRenew = now
      return false
    } catch {
      return true
    }
  }
  // A standby checks the lease every half minute, not every poll.
  if (!leaseUntil && lastRenew && !force && now - lastRenew < STANDBY_CHECK_MS) return false
  let current: { cid?: string; value?: Lease } = {}
  try {
    current = (await agent.com.atproto.repo.getRecord(LEASE)).data as unknown as typeof current
  } catch (e) {
    if (!/not found|could not locate/i.test(String((e as Error).message))) {
      log('lease read failed:', (e as Error).message)
      return now < leaseUntil
    }
  }
  const mine = current.value?.instance === INSTANCE
  const live = current.value && Date.parse(current.value.expiresAt) > now
  if (live && !mine && !force) {
    if (leaseUntil) log(`lease lost to ${current.value!.instance}; standing by`)
    else if (!lastRenew) log(`lease held by ${current.value!.instance} until ${current.value!.expiresAt}; standing by`)
    leaseUntil = 0
    lastRenew = now
    return false
  }
  const record = { $type: LEASE.collection, instance: INSTANCE, expiresAt: new Date(now + LEASE_TTL_MS).toISOString() }
  try {
    if (current.cid) await agent.com.atproto.repo.putRecord({ ...LEASE, record, swapRecord: current.cid })
    else await agent.com.atproto.repo.createRecord({ ...LEASE, record })
  } catch (e) {
    log('lease write failed:', (e as Error).message)
    leaseUntil = 0
    lastRenew = now
    return false
  }
  if (!mine) log(`lease taken by ${INSTANCE}${force && live ? ` (forced from ${current.value!.instance})` : ''}`)
  force = false
  leaseUntil = now + LEASE_TTL_MS
  lastRenew = now
  return true
}

// Hand the lease back on shutdown, so a replacement can start at once.
for (const sig of ['SIGINT', 'SIGTERM'] as const)
  process.once(sig, async () => {
    if (leaseUntil > Date.now())
      await agent.com.atproto.repo.getRecord(LEASE).then(({ data }) =>
        (data.value as Lease).instance === INSTANCE ? agent.com.atproto.repo.deleteRecord({ ...LEASE, swapRecord: data.cid }) : undefined,
      ).catch(() => {})
    log(`stopping (${sig})`)
    process.exit(0)
  })

// --- Loop ----------------------------------------------------------------------
for (;;) {
  if (await holdLease()) {
    for (const [name, poll] of [['mentions', pollMentions], ['dms', pollDms]] as const) {
      try {
        await poll()
      } catch (e) {
        log(`${name} poll failed:`, e instanceof Error ? e.message : e)
      }
    }
    if (firstRun) log('first run: marked existing mentions and DMs as handled')
    firstRun = false
  }
  await new Promise((r) => setTimeout(r, POLL_MS))
}
