#!/usr/bin/env node
import { createInterface } from "node:readline";
import { evaluate } from "./eval_xss.mjs";

function parseArgs(argv) {
  const out = { payload: null };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === "--payload") out.payload = argv[++i];
  }
  return out;
}

async function runStdin() {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  let fallbackId = 0;
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch {
      msg = { payload: trimmed };
    }
    const id = msg.id ?? fallbackId++;
    let result;
    try {
      result = await evaluate(msg.payload ?? "", {
        fullDocument: Boolean(msg.document || msg.fullDocument),
      });
    } catch (err) {
      result = {
        executed: false,
        hooks: [],
        error: String(err).slice(0, 300),
      };
    }
    process.stdout.write(JSON.stringify({ id, ...result }) + "\n");
  }
}

const args = parseArgs(process.argv);
if (args.payload != null) {
  process.stdout.write(JSON.stringify(await evaluate(args.payload)) + "\n");
} else {
  await runStdin();
}
