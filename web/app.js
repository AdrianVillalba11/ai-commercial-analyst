"use strict";
/*
  LÓGICA DE LA PÁGINA

  Qué hace: envía la pregunta a la API, muestra el avance paso a paso y dibuja la respuesta en tarjetas.
  Con qué se conecta: consume los endpoints de src/analyst/api/main.py (/api/ask/stream, /api/health,
  /api/quality, /api/security, /api/evidence/...) y dibuja dentro de web/index.html.
  Idea clave de seguridad: todo lo que viene del servidor (documentos, texto del modelo) se muestra como
  TEXTO PLANO: los nodos se crean con textContent, nunca a partir de cadenas HTML. Así un documento
  malicioso no puede ejecutar nada en la página.

  Bloques: constantes y formato -> ayudantes -> estado -> progreso -> preguntar -> fuentes -> tarjetas
  de la respuesta -> panel lateral -> arranque.
*/

/* =====================================================================
   BLOQUE: Constantes y formato de números
   ===================================================================== */
/* Preguntas de ejemplo: [etiqueta del botón, pregunta completa]. */

const EXAMPLES = [
  ["Región con mayor caída",
   "¿Cuál fue la región con mayor caída porcentual en ventas netas de Brío entre T1 y T2 de 2026? Muestra los valores y el cálculo."],
  ["Hallazgos que explican el deterioro",
   "¿Qué hallazgos de consumidor y observaciones comerciales podrían ayudar a explicar el deterioro de Brío?"],
  ["¿Es consistente el reporte?",
   "¿Son consistentes las principales conclusiones del reporte comercial con los datos de ventas?"],
  ["¿Causaron las promociones la caída?",
   "¿Podemos afirmar que las promociones de la competencia causaron la caída de ventas? ¿Qué evidencia lo sustenta o limita?"],
];
/* Texto que se muestra en cada paso del progreso. */
const STEPS = {
  planificacion: "Planificando la consulta",
  calculo: "Calculando con los datos",
  recuperacion: "Buscando evidencia en los documentos",
  redaccion: "Redactando la respuesta",
  validacion: "Verificando la respuesta",
};
/* Etiquetas legibles para el origen de cada afirmación y el modo de búsqueda. */
const ORIGINS = { calculo: "Cálculo", documento: "Documento", contraste: "Contraste", modelo: "Modelo" };
const MODES = { semantic: "semántica", hybrid: "híbrida", lexical: "léxica" };

/* Atajo para buscar un elemento por su id. */
const $ = (id) => document.getElementById(id);
/* Formatos numéricos colombianos: punto para miles y coma para decimales. */
const intFormat = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 0 });
const pctFormat = new Intl.NumberFormat("es-CO", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtNum = (n) => intFormat.format(n);
const fmtPct = (n) => (n === null || n === undefined ? "—" : `${pctFormat.format(n)} %`);
const fmtPp = (n) => `${n > 0 ? "+" : ""}${pctFormat.format(n)} pp`;
const fmtSec = (s) => `${pctFormat.format(s).replace(",00", "")} s`;

/* =====================================================================
   BLOQUE: Ayudantes
   ===================================================================== */

/* Crea un elemento: clase, texto (con textContent), atributos y hijos. Los hijos nulos se ignoran. */
function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text !== undefined) node.textContent = options.text;
  for (const [name, value] of Object.entries(options.attrs || {})) node.setAttribute(name, value);
  for (const child of children) if (child) node.append(child);
  return node;
}

/* Saca el mensaje de error que envió el servidor; si no hay uno legible, devuelve uno genérico. */
async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (_) { /* not JSON */ }
  return `La aplicación respondió con un error (${response.status}).`;
}

/* GET que devuelve el JSON, o lanza un error con el mensaje del servidor. */
async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* =====================================================================
   BLOQUE: Estado de Ollama (indicador de la barra superior)
   ===================================================================== */

/* Cambia el color y el texto del indicador de estado. */
function setStatus(kind, text, hint) {
  const chip = $("status");
  chip.className = `status status-${kind}`;
  chip.textContent = text;
  chip.title = hint || "";
}

/* Consulta /api/health y actualiza el indicador; se repite sola (cada 2 s mientras calienta, si no cada 20 s). */
async function refreshStatus() {
  let next = 20000;
  try {
    const h = await getJson("/api/health");
    if (!h.running) {
      setStatus("bad", "Ollama apagado", "Abre la aplicación de Ollama o ejecuta: ollama serve");
    } else if (!h.chat_model.available) {
      setStatus("bad", `Falta el modelo ${h.chat_model.name}`, `Ejecuta: ollama pull ${h.chat_model.name}`);
    } else if (!h.embedding_model.available) {
      setStatus("bad", `Falta el modelo ${h.embedding_model.name}`, `Ejecuta: ollama pull ${h.embedding_model.name}`);
    } else if (h.warm.state === "pending") {
      setStatus("wait", "Calentando modelos…", "Cargando los modelos en memoria para que la primera pregunta sea rápida");
      next = 2000;
    } else if (h.warm.state === "error") {
      setStatus("wait", "Modelos sin precargar", `La primera pregunta puede tardar más. ${h.warm.detail}`);
    } else {
      setStatus("ok", `Listo · ${h.chat_model.name} · búsqueda ${MODES[h.retrieval_mode] || h.retrieval_mode}`,
        `${h.documents} fragmentos de documentos y ${fmtNum(h.sales_rows)} filas de ventas limpias`);
    }
  } catch (_) {
    setStatus("bad", "Sin conexión con la aplicación", "Comprueba que `python run.py` sigue ejecutándose");
  }
  setTimeout(refreshStatus, next);
}

/* =====================================================================
   BLOQUE: Progreso mientras se responde
   ===================================================================== */

let timer = null;

/* Muestra la tarjeta de progreso y arranca el contador de segundos. */
function startProgress() {
  $("steps").replaceChildren();
  $("progress").hidden = false;
  const started = performance.now();
  $("elapsed").textContent = "Transcurrido: 0 s";
  timer = setInterval(() => {
    $("elapsed").textContent = `Transcurrido: ${fmtSec((performance.now() - started) / 1000)}`;
  }, 200);
}

/* Marca como hecho el paso anterior y agrega el nuevo como activo. */
function onStep(step) {
  for (const item of $("steps").querySelectorAll("li.active")) item.className = "done";
  $("steps").append(el("li", { class: "active", text: STEPS[step] || step }));
}

/* Detiene el contador y oculta el progreso. */
function stopProgress() {
  clearInterval(timer);
  $("progress").hidden = true;      // se oculta antes de dibujar la respuesta, para que la página no se desplace después
}

/* =====================================================================
   BLOQUE: Preguntar (conexión SSE con el servidor)
   El servidor envía eventos: 'step' (avance), 'result' (respuesta) o 'error'.
   ===================================================================== */

/* Convierte un bloque de texto SSE en {event, data}; devuelve null si no trae datos. */
function parseFrame(frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data.push(line.slice(6));
  }
  return data.length ? { event, data: JSON.parse(data.join("\n")) } : null;
}

/* Envía la pregunta y lee los eventos a medida que llegan; devuelve el resultado final. */
async function streamAsk(question, mode) {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  if (!response.ok || !response.body) throw new Error(await readError(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const parsed = parseFrame(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (!parsed) continue;
      if (parsed.event === "step") onStep(parsed.data.step);
      else if (parsed.event === "result") return parsed.data;
      else if (parsed.event === "error") throw new Error(parsed.data.detail);
    }
  }
  throw new Error("La conexión se cerró antes de terminar la respuesta.");
}

/* Muestra un mensaje de error en el recuadro rojo. */
function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = false;
}

/* Flujo completo al pulsar "Preguntar": valida, muestra progreso, pide, y dibuja la respuesta o el error. */
async function ask() {
  const question = $("question").value.trim();
  $("error").hidden = true;
  if (!question) { showError("Escribe una pregunta."); return; }
  $("ask-btn").disabled = true;
  $("answer").hidden = true;
  startProgress();
  try {
    const result = await streamAsk(question, $("mode").value || null);
    stopProgress();
    renderAnswer(result);
  } catch (error) {
    stopProgress();
    showError(error.message);
  } finally {
    $("ask-btn").disabled = false;
  }
}

/* =====================================================================
   BLOQUE: Fuentes citadas (botones con el id de cada fuente)
   ===================================================================== */

/* Botón con el id de una fuente; al hacer clic se abre. */
function sourceChip(id) {
  const chip = el("button", { class: "chip", text: id, attrs: { type: "button", title: "Ver la fuente" } });
  chip.addEventListener("click", () => openSource(id));
  return chip;
}

/* Si la fuente es un cálculo (CALC-n) o el contraste, salta a su tarjeta; si es un documento, abre su texto original. */
function openSource(id) {
  const isCalc = id.startsWith("CALC-") || id.startsWith("CONTRASTE-");
  if (isCalc) {
    const target = $(id.startsWith("CONTRASTE-") ? "contrast" : `calc-${id}`);
    if (target) {
      if (target.tagName === "DETAILS") target.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    }
    return;
  }
  showEvidence(id);
}

/* =====================================================================
   BLOQUE: Tarjetas de la respuesta
   ===================================================================== */

/* Etiqueta de color con el origen de una afirmación (cálculo, documento, contraste, modelo). */
function badge(origin) {
  return el("span", { class: `badge badge-${origin}`, text: ORIGINS[origin] || origin });
}

/* Tarjeta con una lista de afirmaciones y sus fuentes; devuelve null si la lista está vacía. */
function claimsCard(title, subtitle, claims, cardClass, hypothesis) {
  if (!claims.length) return null;
  const list = el("ul", { class: "claims" });
  for (const claim of claims) {
    const meta = el("div", { class: "claim-meta" });
    if (hypothesis) meta.append(el("span", { class: "badge badge-warn", text: "No demostrada" }));
    meta.append(badge(claim.origen));
    for (const source of claim.fuentes || []) meta.append(sourceChip(source));
    list.append(el("li", {}, el("p", { text: claim.texto }), meta));
  }
  return el("section", { class: `card ${cardClass}` }, el("h3", { text: title }),
    subtitle ? el("p", { class: "subtitle", text: subtitle }) : null, list);
}

/* Tarjeta de resumen; avisa cuando lo redactó el sistema y no el modelo. */
function summaryCard(final) {
  const card = el("section", { class: "card card-summary" }, el("h2", { text: "Resumen" }),
    el("p", { class: "summary", text: final.resumen }));
  if (final.resumen_origen !== "modelo") {
    card.append(el("p", { class: "subtitle", text: "Redactado por el sistema a partir de los cálculos y las verificaciones." }));
  }
  return card;
}

/* Tabla genérica: columnas desde `numericFrom` se alinean a la derecha. Cada fila puede traer un título (tooltip). */
function table(headers, rows, numericFrom, wideLast = false) {
  const head = el("tr", {}, ...headers.map((h, i) => el("th", { class: i >= numericFrom ? "num" : "", text: h })));
  const body = rows.map((row) => {
    const tr = el("tr", { class: row.total ? "total" : "" });
    row.cells.forEach((cell, i) => {
      const td = el("td", { class: i >= numericFrom ? "num" : "" });
      if (cell instanceof Node) td.append(cell); else td.textContent = cell;
      tr.append(td);
    });
    if (row.title) tr.title = row.title;
    return tr;
  });
  return el("div", { class: wideLast ? "table-wrap wide-last" : "table-wrap" },
    el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
}

/* Tabla de contraste entre el reporte comercial y los datos, con su veredicto. */
function contrastCard(contrast) {
  if (!contrast) return null;
  const rows = contrast.rows.map((r) => ({
    cells: [r.region, fmtPct(r.reported), fmtPct(r.computed), fmtPp(r.diff_pp),
      el("span", { class: `badge ${r.coincide ? "badge-ok" : "badge-bad"}`, text: r.coincide ? "Coincide" : "No coincide" })],
  }));
  const card = el("section", { class: "card", attrs: { id: "contrast" } },
    el("h3", { text: "Reporte comercial frente a los datos" }),
    el("p", { class: "subtitle", text: `Variación de ventas T2 frente a T1 por región. Tolerancia: ±${contrast.tolerance_pp} pp (el reporte usa porcentajes enteros).` }),
    table(["Región", "Reporte", "Datos", "Diferencia", "Resultado"], rows, 1));
  card.append(el("p", { text: contrast.verdict }));
  return card;
}

/* Tabla de un cálculo CALC-n; al pasar el cursor por una fila se ve la fórmula. */
function calcCard(block) {
  const result = block.result;
  if (!result || !result.total) return null;
  const rows = result.rows.map((r) => ({
    cells: [Object.values(r.keys).join(" / "), fmtNum(r.t1), fmtNum(r.t2), fmtNum(r.abs_change), fmtPct(r.pct_change),
      r.share_of_total_change_pct === null ? "—" : fmtPct(r.share_of_total_change_pct)],
    title: r.calculation,
  }));
  const t = result.total;
  rows.push({ total: true, cells: ["Total Brío", fmtNum(t.t1), fmtNum(t.t2), fmtNum(t.abs_change), fmtPct(t.pct_change), ""],
    title: t.calculation });
  const card = el("details", { class: "card", attrs: { id: `calc-${block.calc_id}`, open: "" } },
    el("summary", { text: `${block.calc_id} · ${block.title}` }),
    table(["Grupo", "T1", "T2", "Variación", "Variación %", "Aporte al cambio"], rows, 1),
    el("p", { class: "subtitle", text: "Pasa el cursor por una fila para ver la fórmula del cálculo." }));
  return card;
}

/* Lista de los fragmentos de documentos que recibió el modelo. */
function evidenceCard(evidence) {
  if (!evidence.length) return null;
  const row = el("div", { class: "chip-row" });
  for (const item of evidence) row.append(sourceChip(item.chunk_id));
  return el("section", { class: "card" }, el("h3", { text: "Evidencia consultada" }),
    el("p", { class: "subtitle", text: "Fragmentos de documentos que recibió el modelo. Haz clic para leer el texto original." }), row);
}

/* Afirmaciones del modelo que el validador descartó, con su motivo. */
function discardsCard(discards) {
  if (!discards.length) return null;
  const card = el("details", { class: "card" },
    el("summary", { text: `Descartado por el validador (${discards.length})` }),
    el("p", { class: "subtitle", text: "Afirmaciones del modelo que no pasaron las verificaciones. No se muestran como respuesta." }));
  for (const d of discards) {
    card.append(el("div", { class: "discard" }, el("p", { text: d.texto }),
      el("p", { class: "reason", text: `Motivo: ${d.motivo}` }),
      el("p", { class: "subtitle", text: `Sección: ${d.seccion}` })));
  }
  return card;
}

/* Detalle técnico: modelo, búsqueda, plan, tiempos, tokens, filas excluidas y avisos. */
function techCard(r) {
  const planOrigin = r.plan_source === "llm" ? "El modelo" : "Plan por defecto";
  const timings = Object.entries(r.timings).map(([k, v]) => `${k} ${fmtSec(v)}`).join(" · ");
  const notices = [...r.warnings, ...r.final.avisos];
  const dl = el("dl", {},
    el("dt", { text: "Modelo" }), el("dd", { text: r.model }),
    el("dt", { text: "Búsqueda" }), el("dd", { text: MODES[r.mode] || r.mode }),
    el("dt", { text: "Plan" }), el("dd", { text: `${r.plan.tipo} (${planOrigin})` }),
    el("dt", { text: "Tiempos" }), el("dd", { text: timings }),
    el("dt", { text: "Tokens" }), el("dd", { text: r.tokens.prompt ? `${r.tokens.prompt} de entrada · ${r.tokens.salida} de salida` : "—" }),
    el("dt", { text: "Filas excluidas" }), el("dd", { text: r.quality_notes.length
      ? r.quality_notes.map((n) => `${n.registro_id} (${n.rule})`).join(", ") : "ninguna" }));
  const card = el("details", { class: "card" }, el("summary", { text: "Detalle técnico" }), dl);
  if (notices.length) {
    card.append(el("h3", { text: "Avisos" }));
    const list = el("ul");
    for (const n of notices) list.append(el("li", { text: n }));
    card.append(list);
  }
  return card;
}

/* Arma todas las tarjetas de la respuesta en orden y las muestra. */
function renderAnswer(r) {
  const root = $("answer");
  const f = r.final;
  const cards = [
    el("p", { class: "answer-meta", text: `${r.model} · búsqueda ${MODES[r.mode] || r.mode} · ${fmtSec(r.timings.total)}` }),
    summaryCard(f),
    claimsCard("Hechos", "Cifras calculadas con los datos y afirmaciones respaldadas por documentos. Cada fuente se puede abrir.", f.hechos, "card-fact", false),
    contrastCard(r.contrast),
    claimsCard("Contradicciones entre fuentes", null, f.contradicciones, "card-contradiction", false),
    claimsCard("Hipótesis", "Explicaciones plausibles que la evidencia sugiere. No están demostradas.", f.hipotesis, "card-hypothesis", true),
    claimsCard("Información insuficiente", "Lo que no se puede afirmar con los datos y documentos disponibles.", f.insuficiente, "card-gap", false),
    ...r.calc.map(calcCard),
    evidenceCard(r.evidence),
    discardsCard(f.descartes),
    techCard(r),
  ];
  root.replaceChildren(...cards.filter(Boolean));   // una tarjeta ausente es null: no debe convertirse en el texto "null"
  root.hidden = false;
  root.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* =====================================================================
   BLOQUE: Panel lateral (fuente citada, calidad de datos, alertas de seguridad)
   ===================================================================== */

/* Abre el panel con un título y un contenido. */
function openDrawer(title, content) {
  $("drawer-title").textContent = title;
  $("drawer-body").replaceChildren(content);
  $("drawer").hidden = false;
  $("overlay").hidden = false;
  $("drawer-close").focus();
}

/* Cierra el panel. */
function closeDrawer() {
  $("drawer").hidden = true;
  $("overlay").hidden = true;
}

/* Muestra el texto original de un fragmento de documento. */
async function showEvidence(id) {
  openDrawer(id, el("p", { class: "hint", text: "Cargando…" }));
  try {
    const e = await getJson(`/api/evidence/${encodeURIComponent(id)}`);
    openDrawer(e.chunk_id, el("div", {},
      el("dl", {},
        el("dt", { text: "Documento" }), el("dd", { text: e.doc_title }),
        el("dt", { text: "Sección" }), el("dd", { text: e.section || "—" }),
        el("dt", { text: "Página" }), el("dd", { text: e.page ?? "—" }),
        el("dt", { text: "Regiones" }), el("dd", { text: e.regions.length ? e.regions.join(", ") : "ninguna mencionada" })),
      el("pre", { class: "quote", text: e.text })));
  } catch (error) {
    openDrawer(id, el("p", { class: "reason", text: error.message }));
  }
}

/* Muestra el reporte de calidad de datos: cada corrección o exclusión. */
async function showQuality() {
  openDrawer("Calidad de datos", el("p", { class: "hint", text: "Cargando…" }));
  try {
    const q = await getJson("/api/quality");
    const s = q.summary;
    const rows = q.issues.map((i) => ({
      cells: [i.registro_id, i.marca, i.rule, i.field, i.original_value, i.action, i.detail],
    }));
    openDrawer("Calidad de datos", el("div", {},
      el("p", { text: `${fmtNum(s.rows_in)} filas leídas; ${fmtNum(s.rows_out)} usadas; ${s.rows_excluded} excluidas.` }),
      el("p", { class: "subtitle", text: "Cada corrección o exclusión queda registrada. Las filas dudosas no se adivinan: se excluyen." }),
      table(["Registro", "Marca", "Regla", "Campo", "Valor original", "Acción", "Motivo"], rows, 99, true)));
  } catch (error) {
    openDrawer("Calidad de datos", el("p", { class: "reason", text: error.message }));
  }
}

/* Muestra los fragmentos sospechosos (cuarentena o advertencia) como texto plano. */
async function showSecurity() {
  openDrawer("Alertas de seguridad", el("p", { class: "hint", text: "Cargando…" }));
  try {
    const { alerts } = await getJson("/api/security");
    const body = el("div", {}, el("p", { class: "subtitle",
      text: "Fragmentos de los documentos que contienen instrucciones dirigidas a una IA. No se usan para responder y se muestran aquí como texto, sin ejecutarse." }));
    if (!alerts.length) body.append(el("p", { text: "No se detectaron fragmentos sospechosos." }));
    for (const a of alerts) {
      const reasons = el("ul");
      for (const r of a.reasons) reasons.append(el("li", { text: `${r.description}: «${r.evidence}»` }));
      body.append(el("div", { class: "alert" },
        el("h3", { text: `${a.chunk_id} · ${a.doc_title}` }),
        el("p", {}, el("span", { class: `badge ${a.status === "cuarentena" ? "badge-bad" : "badge-warn"}`, text: a.status }),
          ` puntaje ${a.score}`),
        reasons, el("pre", { class: "quote", text: a.excerpt })));
    }
    openDrawer("Alertas de seguridad", body);
  } catch (error) {
    openDrawer("Alertas de seguridad", el("p", { class: "reason", text: error.message }));
  }
}

/* =====================================================================
   BLOQUE: Arranque
   ===================================================================== */

/* Prepara los botones de ejemplo y los eventos, y comienza a consultar el estado. */
function init() {
  const row = $("examples");
  for (const [label, question] of EXAMPLES) {
    const chip = el("button", { class: "chip", text: label, attrs: { type: "button", title: question } });
    chip.addEventListener("click", () => { $("question").value = question; ask(); });
    row.append(chip);
  }
  $("ask-btn").addEventListener("click", ask);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); ask(); }
  });
  $("btn-quality").addEventListener("click", showQuality);
  $("btn-security").addEventListener("click", showSecurity);
  $("drawer-close").addEventListener("click", closeDrawer);
  $("overlay").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
  refreshStatus();
}

init();
