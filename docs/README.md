# Immediate Refill project page

Static research-project page for **When Is Immediate Refill Optimal for Parallel Voting?**

The page follows the evidence order used by strong LLM and embodied-AI project pages such as Qwen, DeepSeek, OpenVLA, Mobile ALOHA, Octo, Diffusion Policy, and RT-1:

1. put Paper, Code, Evidence, and Notes in a visible resource row;
2. identify the research question and show a representative result;
3. make the mechanism inspectable;
4. state the central theorem;
5. separate replay evidence from controlled device evidence;
6. expose the source package and reproduction command;
7. finish with a copyable citation.

The figures and claims are copied from the current manuscript package. The page does not add new performance claims or generated demonstrations.

## Preview locally

From `E:\llm\project-page`:

```powershell
python -m http.server 4173 --bind 127.0.0.1
```

Open `http://127.0.0.1:4173/`.

The page itself is plain HTML, CSS, and JavaScript. No build step or package installation is required.

## Publish with GitHub Pages

Copy the contents of `project-page/` into the root of a GitHub Pages repository, or publish the folder from a branch using the repository's Pages settings. Keep `assets/`, `downloads/`, `app.js`, and `styles.css` beside `index.html` so all links remain relative and the page works without a build step.

The canonical repository is [Shiraikuroko123/immediate-refill](https://github.com/Shiraikuroko123/immediate-refill). The publishable copy is placed under `docs/` so the repository can keep its research archive at the root. Enable GitHub Pages from the `main` branch and the `/docs` folder, then the same `index.html` becomes the Pages entry point.
