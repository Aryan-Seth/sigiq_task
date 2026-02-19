import readline from "node:readline";
import latexToSpeech from "latex-to-speech";

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

for await (const line of rl) {
  const raw = String(line || "").trim();
  if (!raw) {
    continue;
  }
  try {
    const payload = JSON.parse(raw);
    const id = String(payload.id || "");
    const exprs = Array.isArray(payload.exprs)
      ? payload.exprs.map((x) => String(x))
      : [];
    const options =
      payload.options && typeof payload.options === "object"
        ? payload.options
        : {};
    const spoken = await latexToSpeech(exprs, options);
    process.stdout.write(`${JSON.stringify({ id, spoken })}\n`);
  } catch (err) {
    const id =
      raw && raw.startsWith("{")
        ? (() => {
            try {
              return String(JSON.parse(raw).id || "");
            } catch (_e) {
              return "";
            }
          })()
        : "";
    process.stdout.write(
      `${JSON.stringify({ id, error: String(err?.message || err) })}\n`
    );
  }
}
