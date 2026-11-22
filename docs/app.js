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

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const targetId = button.dataset.panel;

    tabButtons.forEach((item) => {
      const active = item === button;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-selected", String(active));
    });

    tabPanels.forEach((panel) => {
      const active = panel.id === targetId;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
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

const closeLightbox = () => {
  if (lightbox?.open) lightbox.close();
};

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = button.querySelector("img")?.alt || "";
    if (lightboxCaption) lightboxCaption.textContent = button.dataset.caption || "";
    lightbox.showModal();
  });
});

lightboxClose?.addEventListener("click", closeLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeLightbox();
});
