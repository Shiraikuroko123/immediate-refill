const chinese = document.documentElement.lang.startsWith("zh");
const archiveQuestion = document.getElementById("archive-question");
const archiveBaseline = document.getElementById("archive-baseline");
if (archiveQuestion && archiveBaseline) {
  const measurements = JSON.parse(document.getElementById("archive-values").textContent);
  const refreshQuestion = () => {
    const [baselineTime, indexedTime] = measurements[archiveQuestion.value][archiveBaseline.value];
    const percent = 100 * (baselineTime - indexedTime) / baselineTime;
    document.getElementById("case-id").textContent = `gsm8k_${archiveQuestion.value}`;
    document.getElementById("case-baseline").textContent = archiveBaseline.selectedOptions[0].textContent;
    document.getElementById("case-base").textContent = `${baselineTime.toFixed(3)} s`;
    document.getElementById("case-index").textContent = `${indexedTime.toFixed(3)} s`;
    document.getElementById("case-saving").textContent = `${percent > 0 ? "+" : ""}${percent.toFixed(2)}%`;
  };
  archiveQuestion.addEventListener("change", refreshQuestion);
  archiveBaseline.addEventListener("change", refreshQuestion);
  archiveQuestion.form.hidden = false;
  refreshQuestion();
}
const tr = (en, zh) => (chinese ? zh : en);
const tabs = [...document.querySelectorAll(".tab-button")];
const panels = [...document.querySelectorAll(".result-panel")];
function selectPanel(index, focus = false) {
  for (const [i, tab] of tabs.entries()) {
    const active = i === index;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panels[i].hidden = !active;
  }
  if (focus) tabs[index].focus();
}
document.querySelector(".tab-list").setAttribute("role", "tablist");
for (const [i, tab] of tabs.entries()) {
  tab.setAttribute("role", "tab");
  tab.setAttribute("aria-controls", panels[i].id);
  panels[i].setAttribute("role", "tabpanel");
  panels[i].tabIndex = 0;
  tab.addEventListener("click", (event) => {
    event.preventDefault();
    selectPanel(i);
  });
  tab.addEventListener("keydown", (event) => {
    const next = {
      ArrowRight: (i + 1) % tabs.length,
      ArrowLeft: (i + tabs.length - 1) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
      " ": i,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectPanel(next, true);
  });
}
selectPanel(
  Math.max(
    0,
    panels.findIndex((panel) => "#" + panel.id === location.hash),
  ),
);
const dialog = document.querySelector("#lightbox");
const frame = dialog.querySelector(".figure-view");
const zoom = dialog.querySelector(".figure-zoom");
let opener = null;
function setZoom(active) {
  frame.classList.toggle("zoom", active);
  zoom.setAttribute("aria-pressed", String(active));
  zoom.textContent = active
    ? tr("Fit figure", "显示全图")
    : tr("Enlarge", "放大");
  frame.scrollTo(0, 0);
}
for (const button of document.querySelectorAll(".figure-open")) {
  if (typeof dialog.showModal !== "function") continue;
  button.hidden = false;
  button.addEventListener("click", () => {
    opener = button;
    const preview = frame.querySelector("img");
    preview.src = button.dataset.lightbox;
    preview.alt = button.parentElement.querySelector("img").alt;
    dialog.querySelector("strong").textContent = button.dataset.caption;
    setZoom(matchMedia("(max-width:720px)").matches);
    dialog.showModal();
  });
}
zoom.addEventListener("click", () =>
  setZoom(!frame.classList.contains("zoom")),
);
dialog
  .querySelector(".lightbox-close")
  .addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => {
  opener?.focus({ preventScroll: true });
  opener = null;
});
dialog.addEventListener("click", (event) => {
  const bounds = dialog.getBoundingClientRect();
  if (
    event.target === dialog &&
    (event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom)
  )
    dialog.close();
});
for (const button of document.querySelectorAll("[data-copy]")) {
  if (!navigator.clipboard?.writeText) continue;
  button.hidden = false;
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(source.textContent);
      button.nextElementSibling.textContent = tr("Copied", "已复制");
      const label = button.dataset.label || button.textContent;
      button.dataset.label = label;
      button.textContent = tr("Copied", "已复制");
      setTimeout(() => {
        button.textContent = label;
      }, 1500);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.nextElementSibling.textContent = tr(
        "Text selected for copying",
        "已选中文字，可以复制",
      );
    }
  });
}
