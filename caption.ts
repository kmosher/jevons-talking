// Image descriptions from Florence-2, run in-process with transformers.js. Each image gets a
// detailed caption plus OCR (people post screenshots), joined into one note. The model only
// ever describes images; it never writes Jev's words.
//
// Generation runs one image at a time: concurrent replies queue for the single slot rather than
// competing for the CPU.
//
// Env: JEV_CAPTION_MODEL (default onnx-community/Florence-2-base-ft), JEV_CAPTION_DTYPE (default
// DEFAULT_DTYPE), JEV_CAPTION_DEVICE (default cpu), JEV_MODEL_CACHE (where weights are downloaded to).
// Usage: node --import tsx caption.ts <image-url>...
import { AutoProcessor, env, Florence2ForConditionalGeneration, RawImage } from '@huggingface/transformers'

const MODEL = process.env.JEV_CAPTION_MODEL ?? 'onnx-community/Florence-2-base-ft'
// Only the decoder is quantized: it's most of the generation time, and q8 there reads text as
// well as fp32 does. A quantized vision encoder or encoder garbles OCR (99% read as 69%).
const DEFAULT_DTYPE = { vision_encoder: 'fp32', encoder_model: 'fp32', embed_tokens: 'fp32', decoder_model_merged: 'q8' }
// A dtype name, or JSON mapping each ONNX module to one.
const DTYPE_ENV = process.env.JEV_CAPTION_DTYPE ?? JSON.stringify(DEFAULT_DTYPE)
const DTYPE = DTYPE_ENV.startsWith('{') ? JSON.parse(DTYPE_ENV) : DTYPE_ENV
const DEVICE = process.env.JEV_CAPTION_DEVICE ?? 'cpu'
if (process.env.JEV_MODEL_CACHE) env.cacheDir = process.env.JEV_MODEL_CACHE

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Florence = { model: any; processor: any }
let loading: Promise<Florence> | undefined
// Starts loading on the first call; later calls share it. A failed load is retried next time.
export function loadCaptioner(): Promise<Florence> {
  loading ??= (async () => {
    const [model, processor] = await Promise.all([
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      Florence2ForConditionalGeneration.from_pretrained(MODEL, { dtype: DTYPE, device: DEVICE } as any),
      AutoProcessor.from_pretrained(MODEL),
    ])
    return { model, processor }
  })()
  loading.catch(() => (loading = undefined))
  return loading
}

let slot: Promise<unknown> = Promise.resolve()
function oneAtATime<T>(job: () => Promise<T>): Promise<T> {
  const result = slot.then(job)
  slot = result.catch(() => {})
  return result
}

async function run({ model, processor }: Florence, image: RawImage, task: string): Promise<string> {
  const inputs = await processor(image, processor.construct_prompts(task))
  const ids = await model.generate({ ...inputs, max_new_tokens: 256 })
  const text = processor.batch_decode(ids, { skip_special_tokens: false })[0]
  const out = processor.post_process_generation(text, task, image.size)[task]
  // Region OCR returns one label per line of text; plain <OCR> would run the lines together.
  const parts: string[] = typeof out === 'string' ? [out] : (out?.labels ?? [])
  return parts
    .map((p) => p.replace(/<\/?s>|<pad>/g, ' ').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .join(' / ')
}

// A description of the image at `url`, with any text in it appended.
export async function describeImage(url: string): Promise<string | null> {
  const image = await RawImage.fromURL(url)
  const florence = await loadCaptioner()
  const [caption, ocr] = await oneAtATime(async () => [
    await run(florence, image, '<MORE_DETAILED_CAPTION>'),
    await run(florence, image, '<OCR_WITH_REGION>'),
  ])
  // Florence reads a stray letter or two off photos with no text in them.
  const text = ocr.length >= 4 ? `text in image: "${ocr}"` : ''
  return [caption, text].filter(Boolean).join(' ') || null
}

if (import.meta.main) {
  let t = performance.now()
  await loadCaptioner()
  console.log(`loaded ${MODEL} (${DTYPE_ENV}, ${DEVICE}) in ${((performance.now() - t) / 1000).toFixed(1)}s`)
  for (const url of process.argv.slice(2)) {
    t = performance.now()
    const d = await describeImage(url)
    console.log(`${((performance.now() - t) / 1000).toFixed(1)}s ${url}\n  ${d}`)
  }
}
