// 用 CDP 驱动无头 Edge 截一张真实渲染图。
// 为什么不用 edge --screenshot：--virtual-time-budget 在流式响应上会提前结束，
// 截图里只有问题气泡、没有答案。这里改成"导航 → 真等 N 秒 → 截图"。
//
//   node scripts/ui_shot.mjs "http://127.0.0.1:8000/?q=年假有几天" out.png 12000
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const url = process.argv[2] || "http://127.0.0.1:8000/";
const out = process.argv[3] || "shot.png";
const waitMs = Number(process.argv[4] || 12000);
const followup = process.argv[5] || ""; // 可选：第一轮结束后再追问一句，用来验证多轮 UI
const port = 9333;
const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";

const edge = spawn(EDGE, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--hide-scrollbars",
  `--remote-debugging-port=${port}`,
  "--window-size=1100,1000",
  "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function targets() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      const list = await res.json();
      const page = list.find((t) => t.type === "page");
      if (page) return page;
    } catch {}
    await sleep(250);
  }
  throw new Error("DevTools 未就绪");
}

const page = await targets();
const ws = new WebSocket(page.webSocketDebuggerUrl);
const pending = new Map();
const errors = [];
let id = 0;

ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
    return;
  }
  if (msg.method === "Runtime.exceptionThrown") {
    errors.push(msg.params.exceptionDetails?.exception?.description || msg.params.exceptionDetails?.text);
  }
  if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
    errors.push(msg.params.args.map((a) => a.value ?? a.description).join(" "));
  }
});

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const mid = ++id;
    pending.set(mid, { resolve, reject });
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

await new Promise((r) => ws.addEventListener("open", r, { once: true }));
await send("Page.enable");
await send("Runtime.enable");
await send("Page.navigate", { url });
await sleep(waitMs);

if (followup) {
  await send("Runtime.evaluate", {
    expression: `(() => {
      const box = document.getElementById("input");
      box.value = ${JSON.stringify(followup)};
      document.getElementById("send").click();
    })()`,
  });
  await sleep(waitMs);
}

const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
writeFileSync(out, Buffer.from(data, "base64"));

const text = await send("Runtime.evaluate", {
  expression: "document.querySelector('.answer')?.innerText.slice(0,120) || '(空)'",
  returnByValue: true,
});
console.log("答案区：", text.result.value.replace(/\n/g, " "));
console.log("JS 错误：", errors.length ? errors.join(" | ").slice(0, 300) : "无");
console.log("截图：", out);

ws.close();
edge.kill();
