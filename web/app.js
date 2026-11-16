const STAGES = {
  queued: "в очереди",
  asr: "расшифровка",
  translate: "перевод",
  write: "запись результата",
  done: "готово",
  error: "ошибка",
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function refresh() {
  try {
    const jobs = await (await fetch("/api/jobs")).json();
    const tbody = document.querySelector("#jobs tbody");
    tbody.innerHTML = "";
    for (const j of jobs) {
      const tr = document.createElement("tr");
      const status = j.stage === "error"
        ? `ошибка: ${j.error}`
        : `${STAGES[j.stage] || j.stage}${j.stage === "asr" ? " " + Math.round(j.progress * 100) + "%" : ""}`;
      tr.innerHTML = `<td>${esc(j.filename)}</td><td>${esc(j.language)}</td>` +
        `<td>${esc(j.model_size)}</td><td>${esc(status)}</td>` +
        `<td>${j.stage === "done" ? `<button data-id="${j.id}">Открыть</button>` : ""}</td>`;
      const btn = tr.querySelector("button");
      if (btn) btn.onclick = () => showResult(j.id, j.filename);
      tbody.appendChild(tr);
    }
  } catch (e) { /* сервис недоступен — просто ждём следующего опроса */ }
}

async function showResult(id, filename) {
  document.getElementById("result").hidden = false;
  document.getElementById("result-name").textContent = filename;
  for (const [elId, file] of [["transcript", "transcript.txt"], ["translation", "translation.txt"]]) {
    const r = await fetch(`/api/jobs/${id}/files/${file}`);
    document.getElementById(elId).textContent = r.ok ? await r.text() : "(нет данных)";
  }
  for (const [linkId, file] of [["dl-transcript", "transcript.txt"],
       ["dl-translation", "translation.txt"], ["dl-docx", "result.docx"]]) {
    document.getElementById(linkId).href = `/api/jobs/${id}/files/${file}`;
  }
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = document.getElementById("upload-status");
  const f = document.getElementById("file").files[0];
  if (!f) return;
  status.textContent = "Загрузка файла…";
  const body = new FormData();
  body.append("file", f);
  body.append("language", document.getElementById("language").value);
  body.append("model_size", document.getElementById("model_size").value);
  const r = await fetch("/api/jobs", { method: "POST", body });
  status.textContent = r.ok
    ? "Файл принят, задача в очереди."
    : `Ошибка: ${r.status} ${await r.text()}`;
  if (r.ok) refresh();
});

refresh();
setInterval(refresh, 2000);
