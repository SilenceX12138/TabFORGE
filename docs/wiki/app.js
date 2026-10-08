// Curated documentation navigation. Existing Markdown routes remain shareable.
const PAGE_GROUPS = [
  {
    "title": "Learn",
    "pages": [
      [
        "overview.md",
        "Home"
      ],
      [
        "getting_started.md",
        "Getting Started"
      ],
      [
        "user_guide.md",
        "User Guide"
      ],
      [
        "tutorials.md",
        "Examples"
      ],
      [
        "cli.md",
        "Command line"
      ],
      [
        "paper.md",
        "Paper \u00b7 NeurIPS 2026"
      ]
    ]
  },
  {
    "title": "Reference",
    "pages": [
      [
        "public_estimator_api.md",
        "API Reference"
      ],
      [
        "public_estimator_api_estimators.md",
        "Estimator classes"
      ],
      [
        "configuration.md",
        "Configuration"
      ],
      [
        "public_estimator_api_lifecycle.md",
        "Lifecycle & checkpoints"
      ]
    ]
  },
  {
    "title": "Inside TabFORGE",
    "pages": [
      [
        "Tabular_representation_pipeline.md",
        "Table preprocessing & embedding"
      ],
      [
        "tabular_feature_processing.md",
        "Feature processing"
      ],
      [
        "tabpfn_embedding_adapter.md",
        "Structure-aware Feature Encoder"
      ],
      [
        "Latent_diffusion_learning_engine.md",
        "Two-stage training & inference"
      ],
      [
        "latent_diffusion_model.md",
        "Score-based Diffusion Transformer & Denoising-aligned Latent Decoder"
      ],
      [
        "training_orchestration.md",
        "Training orchestration"
      ],
      [
        "User-facing_estimator_and_configuration_interface.md",
        "Estimator architecture"
      ],
      [
        "development.md",
        "Development"
      ]
    ]
  }
];
const PAGES = PAGE_GROUPS.flatMap((group) => group.pages);
const DOC_ORDER = PAGES.map(([file]) => file);
const DOCS_BASE_PATH = "";

// Router / rendering state
let currentFile = null;
let loadToken = 0;
let scrollSpyObserver = null;

// Initialize marked. headerIds/mangle were removed in marked v5+;
// GitHub-compatible heading ids come from the marked-gfm-heading-id
// extension (wraps github-slugger), matching the anchors used in
// cross-document links inside the generated docs.
marked.setOptions({
  breaks: false,
  gfm: true,
});
if (typeof markedGfmHeadingId !== "undefined") {
  marked.use(markedGfmHeadingId.gfmHeadingId());
}

if (typeof hljs !== "undefined") {
  hljs.configure({ ignoreUnescapedHTML: true });
  hljs.registerLanguage("bibtex", () => ({
    name: "BibTeX",
    case_insensitive: true,
    contains: [
      hljs.COMMENT("%", "$"),
      { className: "keyword", begin: /@[a-z]+/ },
      { className: "attr", begin: /\b[a-z_]+(?=\s*=)/ },
      hljs.QUOTE_STRING_MODE,
      { className: "string", begin: /\{(?=[^{}]*\})/, end: /\}/ },
      hljs.NUMBER_MODE,
    ],
  }));
}

function isDarkTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark";
}

function initMermaid() {
  const config = {
    startOnLoad: false,
    suppressErrorRendering: true,
    theme: "base",
    flowchart: { htmlLabels: true, curve: "basis" },
    sequence: { mirrorActors: true, useMaxWidth: true },
  };
  config.themeVariables = isDarkTheme() ? {
    darkMode: true,
    primaryColor: "#263857",
    primaryTextColor: "#dbe6ff",
    primaryBorderColor: "#7194ee",
    lineColor: "#9db8ff",
    secondaryColor: "#29313d",
    tertiaryColor: "#14181e",
  } : {
    primaryColor: "#edf3ff",
    primaryTextColor: "#172554",
    primaryBorderColor: "#b5c9ff",
    lineColor: "#3964fe",
    secondaryColor: "#f7f9fc",
    tertiaryColor: "#f5f8ff",
  };
  mermaid.initialize(config);
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", function () {
  setupNavFilter();
  setupThemeToggle();
  setupMobileDrawer();
  setupHeaderControls();
  setupDocumentSearch();
  setupLinkInterception();
  initMermaid();
  document.addEventListener("keydown", (event) => {
    const editing = /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable;
    if ((event.key === "/" && !editing) || ((event.ctrlKey || event.metaKey) && event.key === "k")) {
      event.preventDefault();
      openDocumentSearch();
    }
    if (event.key === "Escape") closeDrawer();
  });

  window.addEventListener("hashchange", onHashChange);
  if (!location.hash) {
    history.replaceState(null, "", buildHash("overview.md"));
  }
  onHashChange();
});

// ---------- Hash routing ----------
// URL scheme: #/<encodeURIComponent(filename)> or
//             #/<encodeURIComponent(filename)>:<heading-slug>
// encodeURIComponent escapes both '&' and ':' in filenames, and GitHub
// slugs never contain ':', so splitting on the first ':' is unambiguous.
// The leading '#/' distinguishes router hashes from plain in-page anchors.

function buildHash(file, anchor) {
  return "#/" + encodeURIComponent(file) + (anchor ? ":" + anchor : "");
}

function parseHash() {
  const hash = location.hash;
  if (hash && hash.indexOf("#/") === 0) {
    const raw = hash.slice(2);
    const sep = raw.indexOf(":");
    const filePart = sep === -1 ? raw : raw.slice(0, sep);
    const anchor = sep === -1 ? null : raw.slice(sep + 1);
    try {
      const file = decodeURIComponent(filePart);
      if (file) {
        return { file: file, anchor: anchor || null };
      }
    } catch (e) {
      // Malformed hash: fall through to the overview
    }
  }
  return { file: "overview.md", anchor: null };
}

function navigateTo(file, anchor) {
  closeDrawer();
  const target = buildHash(file, anchor);
  if (location.hash === target) {
    // Same URL (e.g. clicking the current link again): re-run manually
    onHashChange();
  } else {
    location.hash = target;
  }
}

function onHashChange() {
  closeDrawer();
  const route = parseHash();
  updateDocumentShell(route.file);
  if (route.file !== currentFile) {
    loadDocument(route.file, route.anchor);
  } else if (route.anchor) {
    scrollToAnchor(route.anchor);
  } else {
    window.scrollTo(0, 0);
  }
  setActiveNav(route.file);
}

function updateDocumentShell(filename) {
  document.body.classList.toggle("home-layout", filename === "overview.md");
  document.body.classList.toggle("intro-layout", filename === "getting_started.md");
  const group = PAGE_GROUPS.find((item) => item.pages.some(([file]) => file === filename)) || PAGE_GROUPS[0];
  buildNavigation(group);
  const topPage = group.title === "Reference" ? "public_estimator_api.md"
    : group.title === "Inside TabFORGE" ? "development.md" : filename;
  document.querySelectorAll(".top-links > a").forEach((link) => {
    const active = link.dataset.page === topPage;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.querySelector(".header-more").open = false;
}

function scrollToAnchor(anchor) {
  if (!anchor) return;
  const el = document.getElementById(anchor);
  if (el) {
    el.scrollIntoView({ block: "start" });
  }
}

// ---------- Sidebar navigation ----------

function buildNavigation(group) {
  const nav = document.getElementById("navigation");
  document.getElementById("section-title").textContent = group.title;
  document.getElementById("nav-filter").value = "";
  const links = group.pages.map(([file, title]) =>
    `<div class="nav-leaf"><a class="nav-item" data-file="${file}" href="${buildHash(file)}">${escapeHtml(title)}</a></div>`
  ).join("");
  const sections = PAGE_GROUPS.filter((item) => item !== group).map((item) =>
    `<a class="section-link" href="${buildHash(item.pages[0][0])}">${escapeHtml(item.title)} <span aria-hidden="true">→</span></a>`
  ).join("");
  nav.innerHTML = `<section class="nav-group">${links}</section><nav class="section-links" aria-label="Other documentation sections">${sections}</nav>`;
}

function setActiveNav(filename) {
  document.querySelectorAll(".nav-item").forEach((item) => {
    const isActive = item.getAttribute("data-file") === filename;
    item.classList.toggle("active", isActive);
    if (isActive) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
}

function setupNavFilter() {
  const input = document.getElementById("nav-filter");
  const nav = document.getElementById("navigation");

  input.addEventListener("input", function () {
    const query = this.value.trim().toLowerCase();
    nav.querySelectorAll(".nav-group, .nav-leaf").forEach((node) => {
      const matches = [...node.querySelectorAll(".nav-item")].some((item) =>
        item.textContent.toLowerCase().includes(query)
      );
      node.classList.toggle("nav-hidden", !matches);
    });
  });
}

// ---------- Document loading & rendering ----------

async function loadDocument(filename, anchor) {
  const token = ++loadToken;
  const loading = document.getElementById("loading");
  const content = document.getElementById("content");

  closeMermaidLightbox();

  loading.style.display = "flex";
  content.style.display = "none";

  try {
    const docPath = DOCS_BASE_PATH
      ? DOCS_BASE_PATH + "/" + filename
      : filename;
    const response = await fetch(docPath);
    if (!response.ok) {
      throw new Error("Failed to load " + filename);
    }

    const markdown = await response.text();
    if (token !== loadToken) return; // superseded by a newer navigation

    content.innerHTML = renderMarkdown(markdown);
    loading.style.display = "none";
    content.style.display = "block";
    currentFile = filename;
    document.title = pageTitleFor(filename) + " — TabFORGE";
    document.getElementById("page-label").innerHTML =
      '<a href="#/overview.md" aria-label="Home">⌂</a><span aria-hidden="true">›</span>' +
      '<span aria-current="page">' + escapeHtml(pageTitleFor(filename)) + '</span>';
    document.querySelector(".content-inner").classList.toggle("home-page", filename === "overview.md");

    highlightCodeBlocks();
    addCopyButtons();
    setupCodeTabs();

    await renderMermaidDiagrams();
    if (token !== loadToken) return;

    buildTOC();
    addHeadingLinks(filename);
    renderPagerLinks(filename);

    // Scroll after diagrams render, since they shift the layout
    if (anchor) {
      scrollToAnchor(anchor);
    } else {
      window.scrollTo(0, 0);
    }
  } catch (error) {
    console.error("Error loading document:", error);
    if (token === loadToken) {
      showError("Failed to load document: " + filename);
    }
  }
}

function addHeadingLinks(filename) {
  document.querySelectorAll("#content h1[id], #content h2[id], #content h3[id], #content h4[id]")
    .forEach((heading) => {
      const link = document.createElement("a");
      link.className = "heading-anchor";
      link.href = buildHash(filename, heading.id);
      link.setAttribute("aria-label", "Link to " + heading.textContent);
      link.textContent = "¶";
      heading.appendChild(link);
    });
}

function setupHeaderControls() {
  document.getElementById("sidebar-collapse").onclick = () => {
    document.body.classList.add("sidebar-collapsed");
    document.getElementById("sidebar-expand").focus();
  };
  document.getElementById("sidebar-expand").onclick = () => {
    document.body.classList.remove("sidebar-collapsed");
    document.getElementById("sidebar-collapse").focus();
  };
  document.addEventListener("click", (event) => {
    const more = document.querySelector(".header-more");
    if (!more.contains(event.target)) more.open = false;
  });
}

// ---------- Full-text documentation search ----------

let searchDocuments = [];

function setupDocumentSearch() {
  const dialog = document.getElementById("search-dialog");
  document.getElementById("open-search").onclick = openDocumentSearch;
  document.getElementById("close-search").onclick = () => dialog.close();
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      dialog.close();
    }
  });
  document.getElementById("doc-search").oninput = renderSearchResults;
  document.getElementById("doc-search").onkeydown = (event) => {
    const firstResult = dialog.querySelector(".search-results a");
    if (firstResult && event.key === "Enter") {
      event.preventDefault();
      firstResult.click();
    }
    if (firstResult && event.key === "ArrowDown") {
      event.preventDefault();
      firstResult.focus();
    }
  };
  document.getElementById("search-results").onclick = (event) => {
    if (event.target.closest("a")) dialog.close();
  };
  dialog.addEventListener("close", () => { searchDocuments = []; });
}

async function openDocumentSearch() {
  const dialog = document.getElementById("search-dialog");
  const input = document.getElementById("doc-search");
  if (dialog.open) { input.focus(); return; }
  dialog.showModal();
  input.value = "";
  input.focus();
  document.getElementById("search-results").innerHTML = "";
  document.getElementById("search-status").textContent = "Loading documentation…";
  try {
    searchDocuments = await Promise.all(PAGES.map(async ([file, title]) => {
      const path = DOCS_BASE_PATH ? DOCS_BASE_PATH + "/" + file : file;
      const response = await fetch(path);
      if (!response.ok) throw new Error("Failed to load " + file);
      const markdown = await response.text();

      return { file, title, text: searchableText(markdown) };
    }));
    if (dialog.open) renderSearchResults();
  } catch (error) {
    document.getElementById("search-status").textContent = "Could not load documentation for search.";
    console.error("Documentation search error:", error);
  }
}

function searchableText(markdown) {

  return markdown.replace(/```mermaid[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[#*`|]/g, " ")
    .replace(/\s+/g, " ").trim();
}

function renderSearchResults() {
  const query = document.getElementById("doc-search").value.trim().toLowerCase();
  const results = document.getElementById("search-results");
  const status = document.getElementById("search-status");
  if (!query) {
    results.innerHTML = "";
    status.textContent = "Type to search guides, examples, and the API.";
    return;
  }
  const words = query.split(/\s+/);
  const matches = searchDocuments.filter((doc) =>
    words.every((word) => (doc.title + " " + doc.text).toLowerCase().includes(word))
  ).sort((a, b) => Number(b.title.toLowerCase().includes(query)) - Number(a.title.toLowerCase().includes(query)));
  status.textContent = matches.length ? matches.length + " matching pages" : "No matching pages.";
  results.innerHTML = matches.slice(0, 8).map((doc) => {
    const position = Math.max(0, doc.text.toLowerCase().indexOf(words[0]) - 45);
    const snippet = doc.text.slice(position, position + 180);

    return '<li><a href="' + buildHash(doc.file) + '"><strong>' + escapeHtml(doc.title) +
      '</strong><span>' + (position ? "…" : "") + escapeHtml(snippet) + '…</span></a></li>';
  }).join("");
}

function setupLinkInterception() {
  // One delegated listener covers markdown content, the TOC and
  // any other internal links.
  document.addEventListener("click", function (e) {
    const link = e.target.closest("a");
    if (!link || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    const href = link.getAttribute("href");
    if (!href) return;

    // Router hashes (e.g. pager links): let the browser update the
    // hash and the hashchange handler do the rest.
    if (href.indexOf("#/") === 0) return;

    // Plain in-page anchors (TOC, intra-document links): scroll and
    // keep the URL shareable without triggering a reload.
    if (href.charAt(0) === "#") {
      e.preventDefault();
      const slug = safeDecode(href.slice(1));
      scrollToAnchor(slug);
      document.getElementById("mobile-toc").open = false;
      if (currentFile && slug) {
        history.replaceState(null, "", buildHash(currentFile, slug));
      }
      return;
    }

    // Relative markdown links, optionally with an anchor
    // (e.g. "Other_Module.md#some-section")
    if (/^https?:/i.test(href)) return;
    const mdMatch = href.match(/([^\/]*\.md)(?:#(.*))?$/i);
    if (mdMatch) {
      e.preventDefault();
      e.stopPropagation();
      const filename = safeDecode(mdMatch[1]);
      const anchor = mdMatch[2] ? safeDecode(mdMatch[2]) : null;
      navigateTo(filename, anchor);
    }
  });
}

function safeDecode(text) {
  try {
    return decodeURIComponent(text);
  } catch (e) {
    return text;
  }
}

function renderMarkdown(markdown) {
  // Convert markdown to HTML using marked
  let html = marked.parse(markdown);
  html = html.replace(/<table>/g, '<div class="table-scroll"><table>')
    .replace(/<\/table>/g, '</table></div>');

  // Process mermaid code blocks
  html = html.replace(
    /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
    (match, code) => {
      const decodedCode = code
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&amp;/g, "&")
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
      return '<figure class="diagram-frame"><div class="mermaid">' + decodedCode + "</div></figure>";
    },
  );

  return html;
}

function highlightCodeBlocks() {
  if (typeof hljs === "undefined") return;
  // Only highlight fences with an explicit language tag —
  // auto-detection miscolors untagged output/log blocks.
  document
    .querySelectorAll('#content pre code[class*="language-"]')
    .forEach((el) => {
      const language = [...el.classList].find((name) => name.startsWith("language-"))?.slice(9);
      if (language && hljs.getLanguage(language)) hljs.highlightElement(el);
    });
}

function addCopyButtons() {
  if (!navigator.clipboard) return; // requires a secure context

  document.querySelectorAll("#content pre").forEach((pre) => {
    if (
      pre.closest(".mermaid") ||
      pre.closest(".mermaid-error") ||
      pre.closest(".code-block")
    )
      return;

    // Wrap so the button doesn't scroll with wide code
    const wrapper = document.createElement("div");
    wrapper.className = "code-block";
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    addCodeLanguageLabel(pre, wrapper);

    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      const code = pre.querySelector("code");
      const text = code ? code.innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("copied");
        }, 1500);
      });
    });
    wrapper.appendChild(btn);
  });
}

/**
 * Label a fenced example using its explicit Markdown language.
 * @param {HTMLPreElement} pre
 * @param {HTMLDivElement} wrapper
 */
function addCodeLanguageLabel(pre, wrapper) {
  const code = pre.querySelector("code");
  const language = [...code.classList].find((name) => name.startsWith("language-"));
  if (!language) return;
  const label = document.createElement("span");
  label.className = "code-language";
  label.textContent = language.slice(9);
  wrapper.prepend(label);
}

/** Set up tabbed code alternatives within the loaded document. */
function setupCodeTabs() {
  document.querySelectorAll('.code-tabs [role="tablist"]').forEach((tablist) => {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => selectCodeTab(tabs, index));
      tab.addEventListener("keydown", (event) => {
        const direction = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
        if (!direction) return;
        event.preventDefault();
        const next = (index + direction + tabs.length) % tabs.length;
        selectCodeTab(tabs, next);
        tabs[next].focus();
      });
    });
  });
}

/**
 * Activate one code example and update its accessible tab state.
 * @param {HTMLButtonElement[]} tabs
 * @param {number} index
 */
function selectCodeTab(tabs, index) {
  tabs.forEach((tab, position) => {
    const selected = position === index;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !selected;
  });
}

async function renderMermaidDiagrams() {
  const mermaidElements = document.querySelectorAll("#content .mermaid");

  for (let i = 0; i < mermaidElements.length; i++) {
    const element = mermaidElements[i];
    // Keep the source around so diagrams can re-render on theme change
    if (!element.dataset.src) {
      element.dataset.src = element.textContent;
    }

    try {
      const renderId = "mermaid-svg-" + Date.now() + "-" + i;
      const { svg } = await mermaid.render(renderId, element.dataset.src);
      element.innerHTML = svg;
      makeMermaidZoomable(element);
    } catch (error) {
      console.error("Mermaid rendering error:", error);
      element.innerHTML =
        '<div class="mermaid-error">⚠️ Diagram failed to render' +
        "<details><summary>Show source</summary><pre>" +
        escapeHtml(element.dataset.src) +
        "</pre></details></div>";
    }
  }
}

// ---------- Mermaid full-screen lightbox with pan & zoom ----------

let lightboxEl = null;
const lightboxState = { scale: 1, tx: 0, ty: 0, width: 0, height: 0 };

function makeMermaidZoomable(element) {
  element.classList.add("zoomable");

  // Keep labels at their natural size; the diagram viewport scrolls as needed.
  const svg = element.querySelector("svg");
  svg.style.width = svg.viewBox.baseVal.width + "px";
  svg.style.maxWidth = "none";

  const frame = element.closest(".diagram-frame");
  if (!frame.querySelector(".diagram-tools")) {
    const toolbar = document.createElement("div");
    toolbar.className = "diagram-tools";
    toolbar.innerHTML = '<span>Scroll to explore</span><button class="mermaid-expand" type="button">Expand diagram <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 2H2v5m11-5h5v5M2 13v5h5m6 0h5v-5"/></svg></button>';
    toolbar.querySelector("button").onclick = () => openMermaidLightbox(element);
    frame.prepend(toolbar);
  }

  element.onclick = function (e) {
    if (e.target.closest("a")) return; // keep links inside diagrams working
    openMermaidLightbox(element);
  };
}

function ensureLightbox() {
  if (lightboxEl) return lightboxEl;

  lightboxEl = document.createElement("div");
  lightboxEl.className = "mermaid-lightbox";
  lightboxEl.setAttribute("role", "dialog");
  lightboxEl.setAttribute("aria-label", "Diagram viewer");
  lightboxEl.innerHTML =
    '<div class="lightbox-canvas"><div class="lightbox-stage"></div></div>' +
    '<div class="lightbox-toolbar">' +
    '<button type="button" data-action="zoom-out" title="Zoom out" aria-label="Zoom out">−</button>' +
    '<button type="button" data-action="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>' +
    '<button type="button" data-action="reset" title="Reset view" aria-label="Reset view">Fit</button>' +
    '<button type="button" data-action="close" title="Close (Esc)" aria-label="Close">✕</button>' +
    "</div>" +
    '<div class="lightbox-hint">Scroll to zoom · drag to pan · Esc to close</div>';
  document.body.appendChild(lightboxEl);

  const canvas = lightboxEl.querySelector(".lightbox-canvas");

  lightboxEl
    .querySelector(".lightbox-toolbar")
    .addEventListener("click", function (e) {
      const button = e.target.closest("button");
      if (!button) return;
      const action = button.getAttribute("data-action");
      const cx = canvas.clientWidth / 2;
      const cy = canvas.clientHeight / 2;
      if (action === "zoom-in") zoomLightboxAt(cx, cy, 1.4);
      else if (action === "zoom-out") zoomLightboxAt(cx, cy, 1 / 1.4);
      else if (action === "reset") fitLightbox();
      else if (action === "close") closeMermaidLightbox();
    });

  // Wheel zoom centred on the cursor (trackpad pinch arrives as
  // ctrl+wheel and is covered by the same handler)
  canvas.addEventListener(
    "wheel",
    function (e) {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      zoomLightboxAt(e.clientX - rect.left, e.clientY - rect.top, factor);
    },
    { passive: false },
  );

  // Drag to pan; two pointers pinch to zoom (touch)
  const pointers = new Map();
  let lastPinchDist = 0;

  canvas.addEventListener("pointerdown", function (e) {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) canvas.classList.add("dragging");
    if (pointers.size === 2) {
      const pts = Array.from(pointers.values());
      lastPinchDist = Math.hypot(
        pts[0].x - pts[1].x,
        pts[0].y - pts[1].y,
      );
    }
  });

  canvas.addEventListener("pointermove", function (e) {
    if (!pointers.has(e.pointerId)) return;
    const prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (pointers.size === 1) {
      lightboxState.tx += e.clientX - prev.x;
      lightboxState.ty += e.clientY - prev.y;
      applyLightboxTransform();
    } else if (pointers.size === 2) {
      const pts = Array.from(pointers.values());
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      if (lastPinchDist > 0) {
        const rect = canvas.getBoundingClientRect();
        const midX = (pts[0].x + pts[1].x) / 2 - rect.left;
        const midY = (pts[0].y + pts[1].y) / 2 - rect.top;
        zoomLightboxAt(midX, midY, dist / lastPinchDist);
      }
      lastPinchDist = dist;
    }
  });

  function releasePointer(e) {
    pointers.delete(e.pointerId);
    lastPinchDist = 0;
    if (pointers.size === 0) canvas.classList.remove("dragging");
  }
  canvas.addEventListener("pointerup", releasePointer);
  canvas.addEventListener("pointercancel", releasePointer);

  // Double-click to zoom in on a spot
  canvas.addEventListener("dblclick", function (e) {
    const rect = canvas.getBoundingClientRect();
    zoomLightboxAt(e.clientX - rect.left, e.clientY - rect.top, 2);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && lightboxEl.classList.contains("open")) {
      closeMermaidLightbox();
    }
  });

  return lightboxEl;
}

function openMermaidLightbox(element) {
  const svg = element.querySelector("svg");
  if (!svg) return;

  const lb = ensureLightbox();
  const stage = lb.querySelector(".lightbox-stage");
  stage.innerHTML = "";

  const clone = svg.cloneNode(true);
  let width = 800;
  let height = 600;
  const vb = clone.viewBox && clone.viewBox.baseVal;
  if (vb && vb.width && vb.height) {
    width = vb.width;
    height = vb.height;
  }
  // Give the clone its natural size; scaling happens via transform
  clone.style.maxWidth = "none";
  clone.style.width = width + "px";
  clone.style.height = height + "px";
  clone.setAttribute("width", width);
  clone.setAttribute("height", height);
  stage.style.width = width + "px";
  stage.style.height = height + "px";
  stage.appendChild(clone);

  lightboxState.width = width;
  lightboxState.height = height;

  lb.classList.add("open");
  document.body.style.overflow = "hidden";
  fitLightbox();
}

function fitLightbox() {
  const canvas = lightboxEl.querySelector(".lightbox-canvas");
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  // Fill most of the screen — small diagrams get scaled UP
  const fit = Math.min(
    (cw * 0.9) / lightboxState.width,
    (ch * 0.85) / lightboxState.height,
  );
  lightboxState.scale = Math.min(Math.max(fit, 0.1), 10);
  lightboxState.tx = (cw - lightboxState.width * lightboxState.scale) / 2;
  lightboxState.ty =
    (ch - lightboxState.height * lightboxState.scale) / 2;
  applyLightboxTransform();
}

function zoomLightboxAt(x, y, factor) {
  const newScale = Math.min(
    Math.max(lightboxState.scale * factor, 0.1),
    12,
  );
  const applied = newScale / lightboxState.scale;
  lightboxState.tx = x - (x - lightboxState.tx) * applied;
  lightboxState.ty = y - (y - lightboxState.ty) * applied;
  lightboxState.scale = newScale;
  applyLightboxTransform();
}

function applyLightboxTransform() {
  const stage = lightboxEl.querySelector(".lightbox-stage");
  stage.style.transform =
    "translate(" +
    lightboxState.tx +
    "px, " +
    lightboxState.ty +
    "px) scale(" +
    lightboxState.scale +
    ")";
}

function closeMermaidLightbox() {
  if (!lightboxEl) return;
  lightboxEl.classList.remove("open");
  lightboxEl.querySelector(".lightbox-stage").innerHTML = "";
  document.body.style.overflow = "";
}

// ---------- On-page table of contents ----------

function buildTOC() {
  const toc = document.getElementById("toc");
  const mobileToc = document.getElementById("mobile-toc");
  const mobileLinks = document.getElementById("mobile-toc-links");
  const contentInner = document.querySelector(".content-inner");
  const headings = Array.from(
    document.querySelectorAll("#content h2, #content h3"),
  ).filter((h) => h.id);

  if (scrollSpyObserver) {
    scrollSpyObserver.disconnect();
    scrollSpyObserver = null;
  }

  mobileToc.open = false;
  mobileLinks.innerHTML = "";
  mobileToc.hidden = currentFile === "overview.md" || headings.length < 3;
  if (headings.length < 3 || currentFile === "overview.md") {
    toc.innerHTML = "";
    contentInner.classList.add("no-toc");
    return;
  }

  contentInner.classList.remove("no-toc");
  let html = "";
  headings.forEach((h) => {
    html +=
      '<a class="toc-link toc-' +
      h.tagName.toLowerCase() +
      '" href="#' +
      escapeHtml(h.id) +
      '">' +
      escapeHtml(h.textContent) +
      "</a>";
  });
  toc.innerHTML = '<div class="toc-title">On this page</div>' + html;
  mobileLinks.innerHTML = html;

  setupScrollSpy(headings);
}

function setupScrollSpy(headings) {
  scrollSpyObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = entry.target.id;
        document.querySelectorAll(".toc-link").forEach((a) => {
          a.classList.toggle(
            "active",
            a.getAttribute("href") === "#" + id,
          );
        });
      });
    },
    { rootMargin: "0px 0px -70% 0px" },
  );

  headings.forEach((h) => scrollSpyObserver.observe(h));
}

// ---------- Prev/next pager ----------

function pageTitleFor(filename) {
  const page = PAGES.find(([file]) => file === filename);
  return page ? page[1] : filename.replace(/\.md$/i, "").replace(/_/g, " ");
}

function renderPagerLinks(filename) {
  const pager = document.getElementById("pager");
  const idx = DOC_ORDER.indexOf(filename);

  if (idx === -1) {
    pager.innerHTML = "";
    return;
  }

  const prev = idx > 0 ? DOC_ORDER[idx - 1] : null;
  const next = idx < DOC_ORDER.length - 1 ? DOC_ORDER[idx + 1] : null;

  let html = "";
  if (prev) {
    html +=
      '<a class="pager-link pager-prev" href="' +
      buildHash(prev) +
      '">' +
      '<span class="pager-label">← Previous</span>' +
      '<span class="pager-title">' +
      escapeHtml(pageTitleFor(prev)) +
      "</span></a>";
  }
  if (next) {
    html +=
      '<a class="pager-link pager-next" href="' +
      buildHash(next) +
      '">' +
      '<span class="pager-label">Next →</span>' +
      '<span class="pager-title">' +
      escapeHtml(pageTitleFor(next)) +
      "</span></a>";
  }
  pager.innerHTML = html;
}

// ---------- Theme ----------

function setupThemeToggle() {
  syncThemeAssets();

  document
    .getElementById("theme-toggle")
    .addEventListener("click", async function () {
      const dark = !isDarkTheme();
      document.documentElement.setAttribute(
        "data-theme",
        dark ? "dark" : "light",
      );
      try {
        localStorage.setItem("codewiki-theme", dark ? "dark" : "light");
      } catch (e) {
        // Storage unavailable (private mode etc.) — theme just won't persist
      }

      syncThemeAssets();
      initMermaid();
      closeMermaidLightbox();

      // Re-render diagrams from their stored source in the new theme
      document.querySelectorAll("#content .mermaid").forEach((el) => {
        if (el.dataset.src) el.innerHTML = "";
      });
      await renderMermaidDiagrams();
    });
}

function syncThemeAssets() {
  const dark = isDarkTheme();
  const button = document.getElementById("theme-toggle");
  button.textContent = "◐";
  button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  button.title = button.getAttribute("aria-label");

  const lightCss = document.getElementById("hljs-theme-light");
  const darkCss = document.getElementById("hljs-theme-dark");
  if (lightCss) lightCss.disabled = dark;
  if (darkCss) darkCss.disabled = !dark;
}

// ---------- Mobile drawer ----------

function setupMobileDrawer() {
  document
    .getElementById("menu-toggle")
    .addEventListener("click", function () {
      const sidebar = document.getElementById("sidebar");
      sidebar.classList.toggle("open");
      document.getElementById("menu-toggle").setAttribute("aria-expanded", String(sidebar.classList.contains("open")));
      document
        .getElementById("backdrop")
        .classList.toggle("visible", sidebar.classList.contains("open"));
    });

  document
    .getElementById("backdrop")
    .addEventListener("click", closeDrawer);
}

function closeDrawer() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("menu-toggle").setAttribute("aria-expanded", "false");
  document.getElementById("backdrop").classList.remove("visible");
}

// ---------- Utilities ----------

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showError(message) {
  const loading = document.getElementById("loading");
  const content = document.getElementById("content");

  loading.style.display = "none";
  content.style.display = "block";
  content.innerHTML =
    '<div class="error"><h3>⚠️ Error</h3><p>' +
    escapeHtml(message) +
    "</p></div>";
  document.getElementById("pager").innerHTML = "";
  document.getElementById("toc").innerHTML = "";
  document.getElementById("mobile-toc").hidden = true;
  document.querySelector(".content-inner").classList.add("no-toc");
}
