/**
 * Second Brain — Cloudflare Worker
 * https://github.com/rahilp/second-brain-cloudflare
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";

export interface Env {
  DB: D1Database;
  VECTORIZE: VectorizeIndex;
  AI: Ai;
  AUTH_TOKEN: string;
}

// ─── CORS ─────────────────────────────────────────────────────────────────────

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
};

// ─── Thresholds ───────────────────────────────────────────────────────────────

const DUPLICATE_BLOCK_THRESHOLD = 0.95;
const DUPLICATE_FLAG_THRESHOLD = 0.85;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function isAuthorized(request: Request, env: Env): boolean {
  return request.headers.get("Authorization") === `Bearer ${env.AUTH_TOKEN}`;
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

async function embed(text: string, env: Env): Promise<number[]> {
  const result = (await env.AI.run("@cf/baai/bge-small-en-v1.5" as any, { text: [text] })) as any;
  return result.data[0] as number[];
}

// ─── Duplicate detection ──────────────────────────────────────────────────────

type DuplicateResult =
  | { status: "unique" }
  | { status: "blocked"; matchId: string; score: number }
  | { status: "flagged"; matchId: string; score: number };

async function checkDuplicate(content: string, env: Env): Promise<DuplicateResult> {
  const values = await embed(content.slice(0, 500), env);
  const results = await env.VECTORIZE.query(values, { topK: 1, returnMetadata: "all" });

  if (!results.matches.length) return { status: "unique" };

  const top = results.matches[0];
  const score = top.score;
  const matchId = (top.metadata as any)?.parentId ?? top.id;

  if (score >= DUPLICATE_BLOCK_THRESHOLD) return { status: "blocked", matchId, score };
  if (score >= DUPLICATE_FLAG_THRESHOLD) return { status: "flagged", matchId, score };
  return { status: "unique" };
}

// ─── Chunking ─────────────────────────────────────────────────────────────────

function chunkText(text: string, maxChars = 1600, overlapChars = 200): string[] {
  if (text.length <= maxChars) return [text];

  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    let end = start + maxChars;
    if (end < text.length) {
      const lastPeriod = text.lastIndexOf(".", end);
      const lastNewline = text.lastIndexOf("\n", end);
      const breakPoint = Math.max(lastPeriod, lastNewline);
      if (breakPoint > start + maxChars / 2) end = breakPoint + 1;
    }
    chunks.push(text.slice(start, Math.min(end, text.length)).trim());
    start = end - overlapChars;
  }

  return chunks.filter((c) => c.length > 0);
}

// ─── Store entry (full embed + chunk) ────────────────────────────────────────

async function storeEntry(
  env: Env,
  id: string,
  content: string,
  tags: string[],
  source: string,
  now: number
): Promise<void> {
  const chunks = chunkText(content);

  const vectors = await Promise.all(
    chunks.map(async (chunk, i) => ({
      id: chunks.length === 1 ? id : `${id}-chunk-${i}`,
      values: await embed(chunk, env),
      metadata: {
        content: chunk.slice(0, 512),
        parentId: id,
        chunkIndex: i,
        totalChunks: chunks.length,
        tags,
        source,
        created_at: now,
      },
    }))
  );

  await env.VECTORIZE.upsert(vectors);

  const vectorIds = vectors.map((v) => v.id);
  await env.DB.prepare(
    `UPDATE entries SET vector_ids = ? WHERE id = ?`
  ).bind(JSON.stringify(vectorIds), id).run();
}

// ─── Fallback: locate vectors for pre-migration entries ──────────────────────
// Entries written before the vector_ids column existed have vector_ids = '[]'.
// Query Vectorize by the entry's own content embedding and filter matches whose
// metadata.parentId points back to this entry id.

async function findVectorIdsByParent(
  env: Env,
  id: string,
  content: string
): Promise<string[]> {
  const values = await embed(content.slice(0, 512), env);
  const results = await env.VECTORIZE.query(values, {
    topK: 50,
    returnMetadata: "all",
  });
  return results.matches
    .filter((m) => {
      const parentId = (m.metadata as any)?.parentId;
      return parentId === id || m.id === id;
    })
    .map((m) => m.id);
}

// ─── Append to existing entry ─────────────────────────────────────────────────
// Embed-first / delete-after: capture old vector IDs, write new content to D1,
// upsert the new vector set, then delete only the leftover old IDs. If embed
// fails, the old vectors stay intact instead of leaving the entry vectorless.

async function appendToEntry(
  env: Env,
  id: string,
  existingContent: string,
  addition: string,
  tags: string[],
  source: string
): Promise<void> {
  const timestamp = new Date().toLocaleDateString();
  const separator = `\n\n[Update ${timestamp}]: `;
  const newContent = existingContent + separator + addition;

  // Resolve old vector IDs BEFORE storeEntry overwrites vector_ids in D1.
  // Fall back to a Vectorize parentId scan for pre-migration entries.
  const oldRow = await env.DB.prepare(
    `SELECT vector_ids FROM entries WHERE id = ?`
  ).bind(id).first() as Record<string, any> | null;
  let oldVectorIds: string[] = JSON.parse(oldRow?.vector_ids ?? "[]");
  if (!oldVectorIds.length && existingContent) {
    try {
      oldVectorIds = await findVectorIdsByParent(env, id, existingContent);
    } catch (e) {
      console.error("Vectorize parent scan failed (non-fatal):", e);
    }
  }

  await env.DB.prepare(
    `UPDATE entries SET content = ? WHERE id = ?`
  ).bind(newContent, id).run();

  // Embed + upsert new vectors first. If this throws, the old vectors are
  // still intact and recall keeps working — we just have stale content
  // in D1 vs. vectors, which is far less bad than losing all vectors.
  await storeEntry(env, id, newContent, tags, source, Date.now());

  // Delete only the old IDs that aren't reused by the new vector set
  // (storeEntry uses deterministic chunk IDs, so shared IDs were just upserted).
  const newRow = await env.DB.prepare(
    `SELECT vector_ids FROM entries WHERE id = ?`
  ).bind(id).first() as Record<string, any> | null;
  const newVectorIds: string[] = JSON.parse(newRow?.vector_ids ?? "[]");
  const newSet = new Set(newVectorIds);
  const stale = oldVectorIds.filter((vid) => !newSet.has(vid));
  if (stale.length) {
    try {
      await env.VECTORIZE.deleteByIds(stale);
    } catch (e) {
      console.error("Stale vector cleanup failed (non-fatal):", e);
    }
  }
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

function buildMcpServer(env: Env): McpServer {
  const server = new McpServer({ name: "second-brain", version: "1.0.0" });

  // ── remember ────────────────────────────────────────────────────────────
  server.tool(
    "remember",
    "Store an idea, task, or note in your second brain",
    {
      content: z.string().describe("The idea, task, or note to store"),
      tags: z.array(z.string()).optional().describe("Optional tags for filtering"),
      source: z.string().optional().describe("Origin: phone, browser, voice, claude"),
    },
    async ({ content, tags, source }) => {
      const c = content.trim();
      const t = tags ?? [];
      const s = source ?? "claude";

      const dup = await checkDuplicate(c, env);

      if (dup.status === "blocked") {
        return {
          content: [{
            type: "text",
            text: `Duplicate detected (${(dup.score * 100).toFixed(0)}% match) — not stored. Existing entry ID: ${dup.matchId}`,
          }],
        };
      }

      const id = crypto.randomUUID();
      const now = Date.now();
      const finalTags = dup.status === "flagged" ? [...t, "duplicate-candidate"] : t;

      await env.DB.prepare(
        `INSERT INTO entries (id, content, tags, source, created_at) VALUES (?, ?, ?, ?, ?)`
      ).bind(id, c, JSON.stringify(finalTags), s, now).run();

      try {
        await storeEntry(env, id, c, finalTags, s, now);
      } catch (e) {
        console.error("Vectorize insert failed (non-fatal):", e);
      }

      if (dup.status === "flagged") {
        return {
          content: [{
            type: "text",
            text: `Stored with ID: ${id} — note: similar entry exists (${(dup.score * 100).toFixed(0)}% match, ID: ${dup.matchId}). Tagged as duplicate-candidate.`,
          }],
        };
      }

      return { content: [{ type: "text", text: `Stored. ID: ${id}` }] };
    }
  );

  // ── append ───────────────────────────────────────────────────────────────
  server.tool(
    "append",
    "Append new information to an existing entry in your second brain. Use this when something has changed or you have an update to a stored note — preserves the original and adds the update with a timestamp.",
    {
      id: z.string().describe("Entry ID to append to — from recall or list_recent"),
      addition: z.string().describe("The new information to add to the existing entry"),
    },
    async ({ id, addition }) => {
      // Look up the existing entry
      const row = await env.DB.prepare(
        `SELECT id, content, tags, source FROM entries WHERE id = ?`
      ).bind(id).first() as Record<string, any> | null;

      if (!row) {
        return {
          content: [{ type: "text", text: `No entry found with ID: ${id}` }],
        };
      }

      const existingContent = row.content as string;
      const tags: string[] = JSON.parse(row.tags ?? "[]");
      const source = row.source as string;
      const a = addition.trim();

      if (!a) {
        return {
          content: [{ type: "text", text: "Addition cannot be empty." }],
        };
      }

      try {
        await appendToEntry(env, id, existingContent, a, tags, source);
      } catch (e) {
        console.error("Append failed:", e);
        return {
          content: [{ type: "text", text: `Append failed: ${(e as Error).message}` }],
        };
      }

      return {
        content: [{
          type: "text",
          text: `Appended to entry ${id}. The original content is preserved and your update has been added with today's date.`,
        }],
      };
    }
  );

  // ── recall ───────────────────────────────────────────────────────────────
  server.tool(
    "recall",
    "Semantically search your second brain for relevant notes",
    {
      query: z.string().describe("Natural language search query"),
      topK: z.number().int().min(1).max(20).default(5).describe("Number of results"),
      tag: z.string().optional().describe("Filter by a specific tag"),
    },
    async ({ query, topK, tag }) => {
      const values = await embed(query, env);
      const results = await env.VECTORIZE.query(values, {
        topK: topK * 3,
        filter: tag ? { tags: { $eq: tag } } : undefined,
        returnMetadata: "all",
      });

      if (!results.matches.length) {
        return { content: [{ type: "text", text: "Nothing found matching that query." }] };
      }

      const seen = new Set<string>();
      const deduped = results.matches.filter((m) => {
        const parentId = (m.metadata as any)?.parentId ?? m.id;
        if (seen.has(parentId)) return false;
        seen.add(parentId);
        return true;
      }).slice(0, topK);

      const text = deduped.map((m, i) => {
        const meta = m.metadata as Record<string, any>;
        const date = meta?.created_at ? new Date(meta.created_at as number).toLocaleDateString() : "?";
        const tagList = Array.isArray(meta?.tags) && meta.tags.length ? ` [${(meta.tags as string[]).join(", ")}]` : "";
        const src = meta?.source ? ` · ${meta.source}` : "";
        const score = (m.score * 100).toFixed(0);
        const chunkLabel = meta?.totalChunks > 1 ? ` (chunk ${meta.chunkIndex + 1}/${meta.totalChunks})` : "";
        const updateLabel = meta?.isUpdate ? " [updated]" : "";
        return `${i + 1}. [${date}${src}${tagList}] (${score}% match)${chunkLabel}${updateLabel}\n${meta?.content ?? ""}`;
      }).join("\n\n");

      return { content: [{ type: "text", text }] };
    }
  );

  // ── list_recent ──────────────────────────────────────────────────────────
  server.tool(
    "list_recent",
    "List the most recent entries from your second brain",
    {
      n: z.number().int().min(1).max(50).default(10),
      tag: z.string().optional(),
    },
    async ({ n, tag }) => {
      let q = `SELECT id, content, tags, source, created_at FROM entries`;
      const p: (string | number)[] = [];
      if (tag) { q += ` WHERE tags LIKE ?`; p.push(`%"${tag}"%`); }
      q += ` ORDER BY created_at DESC LIMIT ?`; p.push(n);

      const { results } = await env.DB.prepare(q).bind(...p).all();

      if (!results.length) {
        return { content: [{ type: "text", text: "No entries found." }] };
      }

      const text = (results as Record<string, any>[]).map((row, i) => {
        const date = new Date(row.created_at as number).toLocaleDateString();
        const tags: string[] = JSON.parse(row.tags ?? "[]");
        const tagStr = tags.length ? ` · ${tags.join(", ")}` : "";
        return `${i + 1}. [${date} · ${row.source}${tagStr}]\nID: ${row.id as string}\n${row.content}`;
      }).join("\n\n");

      return { content: [{ type: "text", text }] };
    }
  );

  // ── forget ───────────────────────────────────────────────────────────────
  server.tool(
    "forget",
    "Delete an entry from your second brain by ID",
    { id: z.string().describe("Entry ID from recall or list_recent") },
    async ({ id }) => {
      const row = await env.DB.prepare(
        `SELECT content, vector_ids FROM entries WHERE id = ?`
      ).bind(id).first() as Record<string, any> | null;
      let vectorIds: string[] = row ? JSON.parse(row.vector_ids ?? "[]") : [];

      // Fallback for pre-migration entries: vector_ids was never populated,
      // so locate vectors via parentId metadata before we lose the content.
      if (row && !vectorIds.length && row.content) {
        try {
          vectorIds = await findVectorIdsByParent(env, id, row.content as string);
        } catch (e) {
          console.error("Vectorize parent scan failed (non-fatal):", e);
        }
      }

      await env.DB.prepare(`DELETE FROM entries WHERE id = ?`).bind(id).run();

      if (vectorIds.length) {
        try {
          await env.VECTORIZE.deleteByIds(vectorIds);
        } catch (e) {
          console.error("Vectorize delete failed (non-fatal):", e);
        }
      }

      return { content: [{ type: "text", text: `Deleted entry ${id}` }] };
    }
  );

  return server;
}

// ─── Main handler ─────────────────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // POST /capture
    if (url.pathname === "/capture" && request.method === "POST") {
      if (!isAuthorized(request, env)) return json({ error: "Unauthorized" }, 401);

      let body: { content?: string; tags?: string[]; source?: string };
      try { body = await request.json(); } catch { return json({ error: "Invalid JSON" }, 400); }
      if (!body.content?.trim()) return json({ error: "content is required" }, 400);

      const c = body.content.trim();
      const t = body.tags ?? [];
      const s = body.source ?? "api";

      const dup = await checkDuplicate(c, env);

      if (dup.status === "blocked") {
        return json({
          ok: false,
          duplicate: true,
          matchId: dup.matchId,
          score: parseFloat((dup.score * 100).toFixed(1)),
          message: "Near-exact duplicate detected — not stored",
        });
      }

      const id = crypto.randomUUID();
      const now = Date.now();
      const finalTags = dup.status === "flagged" ? [...t, "duplicate-candidate"] : t;

      await env.DB.prepare(
        `INSERT INTO entries (id, content, tags, source, created_at) VALUES (?, ?, ?, ?, ?)`
      ).bind(id, c, JSON.stringify(finalTags), s, now).run();

      try {
        await storeEntry(env, id, c, finalTags, s, now);
      } catch (e) {
        console.error("Vectorize insert failed (non-fatal):", e);
      }

      if (dup.status === "flagged") {
        return json({
          ok: true,
          id,
          warning: "similar",
          matchId: dup.matchId,
          score: parseFloat((dup.score * 100).toFixed(1)),
          message: "Stored but similar entry exists — tagged as duplicate-candidate",
        });
      }

      return json({ ok: true, id });
    }

    // GET /list
    if (url.pathname === "/list" && request.method === "GET") {
      if (!isAuthorized(request, env)) return json({ error: "Unauthorized" }, 401);
      const n = Math.min(parseInt(url.searchParams.get("n") ?? "20", 10), 100);
      const { results } = await env.DB.prepare(
        `SELECT id, content, tags, source, created_at FROM entries ORDER BY created_at DESC LIMIT ?`
      ).bind(n).all();
      return json(results);
    }

    // /mcp
    if (url.pathname === "/mcp") {
      const transport = new WebStandardStreamableHTTPServerTransport({
        sessionIdGenerator: undefined,
      });
      const server = buildMcpServer(env);
      await server.connect(transport);
      return transport.handleRequest(request);
    }

    return new Response("Not found", { status: 404 });
  },
};