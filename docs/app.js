const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector("#site-nav");

if (menuToggle && siteNav) {
  menuToggle.addEventListener("click", () => {
    const isOpen = siteNav.classList.toggle("is-open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
  });

  siteNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      siteNav.classList.remove("is-open");
      menuToggle.setAttribute("aria-expanded", "false");
    });
  });
}

const tabButtons = [...document.querySelectorAll(".tab-button")];
const tabPanels = [...document.querySelectorAll(".tab-panel")];

function selectTab(button) {
  tabButtons.forEach((item) => {
    const active = item === button;
    item.classList.toggle("is-active", active);
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
  });
  tabPanels.forEach((panel) => {
    const active = panel.id === button.dataset.panel;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

tabButtons.forEach((button, index) => {
  button.addEventListener("click", () => selectTab(button));
  button.addEventListener("keydown", (event) => {
    const count = tabButtons.length;
    const target = {
      ArrowLeft: (index + count - 1) % count,
      ArrowRight: (index + 1) % count,
      Home: 0,
      End: count - 1
    }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    selectTab(tabButtons[target]);
    tabButtons[target].focus();
  });
});

const revealItems = document.querySelectorAll(".reveal");
const revealObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

revealItems.forEach((item) => revealObserver.observe(item));

const counterItems = document.querySelectorAll("[data-count]");
const formatNumber = (value) => {
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  return `${value.toFixed(2)}%`;
};

const animateCounter = (element) => {
  const target = element.dataset.count;
  if (target === "iff") return;

  const number = Number(target);
  const duration = 900;
  const start = performance.now();

  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = target === "200"
      ? formatNumber(Math.round(number * eased))
      : formatNumber(number * eased);
    if (progress < 1) requestAnimationFrame(tick);
  };

  requestAnimationFrame(tick);
};

const counterObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.8 }
);

counterItems.forEach((item) => counterObserver.observe(item));

const copyButtons = document.querySelectorAll(".copy-button");
copyButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    const text = target.innerText;
    try {
      await navigator.clipboard.writeText(text);
      const oldText = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = oldText;
      }, 1400);
    } catch {
      button.textContent = "Select text";
    }

    if (target.classList.contains("citation-code")) {
      target.classList.add("is-visible");
    }
  });
});

const lightbox = document.querySelector("#lightbox");
const lightboxImage = document.querySelector("#lightbox-image");
const lightboxCaption = document.querySelector("#lightbox-caption");
const lightboxClose = document.querySelector(".lightbox-close");
const figureViewport = document.querySelector("#figure-viewport");
const figureZoom = document.querySelector(".figure-zoom");
let openedFrom = null;

function setZoom(zoomed) {
  figureViewport.classList.toggle("is-zoomed", zoomed);
  figureZoom.textContent = zoomed ? "Fit figure" : "Enlarge";
  figureViewport.scrollTo(0, 0);
}

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    openedFrom = button;
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = button.querySelector("img")?.alt || "";
    lightboxCaption.textContent = button.dataset.caption || "";
    lightbox.showModal();
    // A narrow screen opens a readable detail view. "Fit figure" shows all of it.
    setZoom(matchMedia("(max-width: 680px)").matches);
  });
});

figureZoom.addEventListener("click", () => setZoom(!figureViewport.classList.contains("is-zoomed")));
lightboxClose.addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target !== lightbox) return;
  const bounds = lightbox.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom) lightbox.close();
});
lightbox.addEventListener("close", () => {
  openedFrom?.focus({preventScroll: true});
  openedFrom = null;
});
