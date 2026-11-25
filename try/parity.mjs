// node try/parity.mjs [parity.json] [model.json]
// The JavaScript port must give the answers Python gave for the same inputs.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { checkParity } from "./airpoint.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(process.argv[2] || join(here, "parity.json"), "utf8"));
const model = JSON.parse(readFileSync(process.argv[3] || join(here, "model.json"), "utf8"));
const { checks, failures } = checkParity(model, fixture);
for (const f of failures.slice(0, 20)) console.log("MISMATCH", f);
console.log(`${checks - failures.length} of ${checks} checks agree with Python`);
process.exit(failures.length ? 1 : 0);
