(() => {
  if (customElements.get("refill-schedule")) return;
  const work = [2, 1, 1];
  const cost = [0, 10, 11, 13];
  const axisMax = 24;
  const labels = {
    en: {
      heading: "Wall-clock replay", costs: "Cost at widths 1/2/3", work: "Path work",
      immediate: "Immediate refill", delayed: "Delay path 3",
      play: "Play", pause: "Pause", replay: "Replay", reset: "Reset",
      time: "Wall time, in model clock units",
      value: (t, end) => `${t} of ${end} wall units`,
      path: (i, start, end) => `Path ${i} runs from ${start} to ${end} wall units.`
    },
    zh: {
      heading: "墙钟时间回放", costs: "并发数为 1/2/3 时的单步耗时", work: "路径步数",
      immediate: "立即补发", delayed: "延后启动路径 3",
      play: "播放", pause: "暂停", replay: "重播", reset: "重置",
      time: "墙钟时间（模型时间单位）",
      value: (t, end) => `当前 ${t}，共 ${end} 个墙钟时间单位`,
      path: (i, start, end) => `路径 ${i} 从墙钟时刻 ${start} 运行至 ${end}。`
    }
  };

  // Virtual steps are synchronized. Convert each step to wall time using its active width.
  function schedule(starts) {
    let wall = 0;
    const lanes = work.map(() => []);
    const durations = [];
    const steps = Math.max(...work.map((length, i) => starts[i] + length));
    for (let step = 0; step < steps; step++) {
      const active = work.map((length, i) => starts[i] <= step && step < starts[i] + length);
      const duration = cost[active.filter(Boolean).length];
      active.forEach((running, i) => {
        if (running) lanes[i].push({start: wall, end: wall + duration});
      });
      durations.push(duration);
      wall += duration;
    }
    return {lanes, durations, wall};
  }

  const plans = [
    {key: "immediate", ...schedule([0, 0, 0])},
    {key: "delayed", ...schedule([0, 0, 1])}
  ];
  const end = Math.max(...plans.map(plan => plan.wall));
  const pct = t => `${100 * t / axisMax}%`;

  function lanesMarkup(plan, text) {
    return plan.lanes.map((units, i) => {
      const blocks = units.map(({start, end}) =>
        `<span class="rs-unit" data-start="${start}" data-end="${end}" style="left:${pct(start)};width:${pct(end-start)}"></span>`
      ).join("");
      return `<div class="rs-lane">
        <span class="rs-path">P${i+1}</span>
        <div class="rs-track" aria-hidden="true">${blocks}</div>
        <span class="rs-sr">${text.path(i+1, units[0].start, units[units.length-1].end)}</span>
      </div>`;
    }).join("");
  }

  function chartMarkup(plan, text) {
    return `<section class="rs-arm rs-${plan.key}" aria-label="${text[plan.key]}">
      <div class="rs-arm-title"><strong>${text[plan.key]}</strong><span>${plan.durations.join(" + ")} = <b>${plan.wall}</b></span></div>
      <div class="rs-lanes">${lanesMarkup(plan, text)}
        <div class="rs-guide" aria-hidden="true"><i style="left:${pct(plan.wall)}"></i><span></span></div>
      </div>
    </section>`;
  }

  class RefillSchedule extends HTMLElement {
    connectedCallback() {
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
          if (this.isConnected) this.mount();
        }, {once: true});
      } else {
        this.mount();
      }
    }

    mount() {
      if (this.controls) return;
      this.clock = 0;
      this.playing = false;
      this.reduced = matchMedia("(prefers-reduced-motion: reduce)");
      this.text = /^zh(?:-|$)/i.test(document.documentElement.lang) ? labels.zh : labels.en;
      const text = this.text;
      this.innerHTML = `<div class="rs-frame">
        <div class="rs-head"><span>${text.heading}</span><span>${text.costs} <b>10 / 11 / 13</b></span></div>
        <div class="rs-paper">
          <div class="rs-inputs">${text.work} <span>P1 <b>2</b></span><span>P2 <b>1</b></span><span>P3 <b>1</b></span></div>
          ${plans.map(plan => chartMarkup(plan, text)).join("")}
          <div class="rs-axis" aria-hidden="true"><div>${[0, 6, 12, 18, 24].map(t=>`<span style="left:${pct(t)}">${t}</span>`).join("")}</div></div>
        </div>
        <div class="rs-controls">
          <button type="button" class="rs-play">${text.play}</button>
          <label class="rs-slider"><span class="rs-sr">${text.time}</span>
            <input type="range" min="0" max="${end}" step="1" value="0"></label>
          <output class="rs-value" aria-live="off">0 / ${end}</output>
          <button type="button" class="rs-reset">${text.reset}</button>
        </div>
      </div>`;
      this.controls = this.querySelector(".rs-controls");
      this.play = this.querySelector(".rs-play");
      this.slider = this.querySelector("input");
      this.display = this.querySelector("output");
      this.units = [...this.querySelectorAll(".rs-unit")];
      this.play.addEventListener("click", () => this.playing ? this.stop() : this.start());
      this.slider.addEventListener("input", () => {
        this.stop();
        this.render(Number(this.slider.value));
      });
      this.querySelector(".rs-reset").addEventListener("click", () => {
        this.stop();
        this.render(0);
      });
      this.render(0);
    }

    disconnectedCallback() {
      this.stop();
    }

    render(t) {
      this.clock = Math.max(0, Math.min(end, t));
      const whole = Math.floor(this.clock);
      if (!this.playing) this.play.textContent = this.clock >= end ? this.text.replay : this.text.play;
      this.style.setProperty("--rs-cursor", pct(this.clock));
      this.slider.value = whole;
      this.slider.setAttribute("aria-valuetext", this.text.value(whole, end));
      this.display.textContent = `${whole} / ${end}`;
      this.units.forEach(unit => {
        const start = Number(unit.dataset.start);
        const stop = Number(unit.dataset.end);
        const progress = Math.max(0, Math.min(1, (this.clock-start)/(stop-start)));
        unit.style.setProperty("--rs-progress", `${100 * progress}%`);
      });
    }

    stop() {
      cancelAnimationFrame(this.frame);
      this.playing = false;
      if (this.play) this.play.textContent = this.clock >= end ? this.text.replay : this.text.play;
    }

    start() {
      if (this.clock >= end) this.render(0);
      this.playing = true;
      this.play.textContent = this.text.pause;
      const initial = this.clock;
      const started = performance.now();
      const tick = now => {
        const t = Math.min(end, initial + (now-started)/240);
        // In reduced-motion mode change at discrete clock ticks instead of sweeping.
        const shown = this.reduced.matches ? Math.floor(t) : t;
        if (shown !== this.clock) this.render(shown);
        if (t >= end) this.stop();
        else this.frame = requestAnimationFrame(tick);
      };
      this.frame = requestAnimationFrame(tick);
    }
  }
  customElements.define("refill-schedule", RefillSchedule);
})();
