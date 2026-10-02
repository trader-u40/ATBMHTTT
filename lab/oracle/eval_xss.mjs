import { JSDOM, ResourceLoader, VirtualConsole } from "jsdom";

class OfflineLoader extends ResourceLoader {
  fetch() {
    return null;
  }
}

function wrap(payload) {
  return `<!DOCTYPE html><html><head><title>lab</title></head><body>${payload}</body></html>`;
}

function runJavascriptUrl(window, value) {
  if (!value) return;
  const trimmed = String(value).trim();
  const match = trimmed.match(/^javascript:(.*)$/i);
  if (!match) return;
  try {
    (window.__rawEval || window.eval)(match[1]);
  } catch {
    /* payload threw — still counted if a hook already fired */
  }
}

function triggerLegacyHandlers(window) {
  const doc = window.document;
  for (const el of doc.querySelectorAll("[href], [src], [action]")) {
    for (const attr of ["href", "src", "action"]) {
      runJavascriptUrl(window, el.getAttribute(attr));
    }
  }
  for (const img of doc.querySelectorAll("img")) {
    const src = img.getAttribute("src") || "";
    if (/^javascript:/i.test(src)) {
      runJavascriptUrl(window, src);
    }
    try {
      img.dispatchEvent(new window.Event("error"));
    } catch {
      /* ignore */
    }
  }
  for (const el of doc.querySelectorAll("*")) {
    for (const attr of Array.from(el.attributes)) {
      const name = attr.name.toLowerCase();
      if (!name.startsWith("on") || name.length < 3) continue;
      const type = name.slice(2);
      if (type === "load" || (type === "error" && el.tagName === "IMG")) continue;
      try {
        el.dispatchEvent(new window.Event(type, { bubbles: false }));
      } catch {
        /* ignore */
      }
    }
  }
}

function installHooks(window, hits) {
  const mark = (name, extra) => {
    hits.push({
      name,
      extra: extra == null ? null : String(extra).slice(0, 240),
    });
    try {
      if (SINKS.has(name)) window.__XSS_ORACLE = true;
    } catch {
      /* ignore */
    }
  };

  window.alert = (m) => mark("alert", m);
  window.confirm = (m) => {
    mark("confirm", m);
    return true;
  };
  window.prompt = (m) => {
    mark("prompt", m);
    return "x";
  };
  window.print = () => mark("print");

  try {
    const evalFn = window.eval.bind(window);
    Object.defineProperty(window, "__rawEval", { value: evalFn });
    window.eval = (code) => {
      mark("eval", code);
      return evalFn(code);
    };
  } catch {
    /* ignore */
  }

  try {
    const OrigFunction = window.Function;
    const Wrapped = function Function(...args) {
      mark("Function", args.at(-1));
      return OrigFunction.apply(this, args);
    };
    Wrapped.prototype = OrigFunction.prototype;
    window.Function = Wrapped;
  } catch {
    /* ignore */
  }

  try {
    let title = window.document.title;
    Object.defineProperty(window.document, "title", {
      configurable: true,
      get() {
        return title;
      },
      set(v) {
        title = String(v);
        if (title === "XSS_ORACLE") mark("title", title);
      },
    });
  } catch {
    /* ignore */
  }

  try {
    Object.defineProperty(window.document, "cookie", {
      configurable: true,
      get() {
        return "";
      },
      set(v) {
        mark("cookie", v);
      },
    });
  } catch {
    /* ignore */
  }
}

// eval/Function are recorded for tracing but are not proof of XSS: JSDOM itself
// compiles every inline on* handler through window.Function.
const SINKS = new Set(["alert", "confirm", "prompt", "print", "title", "cookie"]);

function waitForLoad(window, ms) {
  return new Promise((resolve) => {
    if (window.document.readyState === "complete") return resolve();
    const timer = setTimeout(resolve, ms);
    window.addEventListener("load", () => {
      clearTimeout(timer);
      setTimeout(resolve, 0);
    });
  });
}

export async function evaluate(payload, options = {}) {
  const hits = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", () => {});
  virtualConsole.on("error", () => {});
  virtualConsole.on("warn", () => {});
  const html = options.fullDocument
    ? String(payload ?? "")
    : wrap(String(payload ?? ""));
  const executed = () => hits.some((h) => SINKS.has(h.name));

  let dom;
  try {
    dom = new JSDOM(html, {
      url: "http://lab.local/page",
      runScripts: "dangerously",
      resources: new OfflineLoader(),
      pretendToBeVisual: true,
      virtualConsole,
      beforeParse(window) {
        installHooks(window, hits);
      },
    });
  } catch (err) {
    return { executed: executed(), hooks: hits, error: String(err).slice(0, 300) };
  }

  try {
    await waitForLoad(dom.window, options.loadTimeoutMs ?? 500);
    triggerLegacyHandlers(dom.window);
    return { executed: executed(), hooks: hits };
  } finally {
    try {
      dom.window.close();
    } catch {
      /* ignore */
    }
  }
}
