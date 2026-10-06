# The Bluesky bot. Runs from the state volume mounted at /data (bot-state.json, transcripts/,
# downloaded model weights), so a restart picks up where it left off.
FROM node:26-slim
# The trace image draws text with system fonts, and the slim image has none: without these, every
# label rendered blank. Liberation matches Arial/Times/Courier metrics, which the layout assumes;
# DejaVu covers symbols Liberation lacks (⌫, ✓).
RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation2 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY *.ts *.txt ./
COPY data ./data
ENV NODE_ENV=production \
    JEV_MODEL_CACHE=/data/models \
    JEV_NORVIG_PAIRS=/data/cache/count_2w.txt
USER node
WORKDIR /data
CMD ["node", "/app/bot.ts"]
