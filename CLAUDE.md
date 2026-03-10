# CLAUDE.md

This file provides guidance to AI assistants (like Claude) working on this repository.

## Project Overview

This is **mehty.github.io**, a personal portfolio website for Amirmehdi Sharifzad (data scientist / ML engineer). It is a static site built with **Jekyll** and hosted on **GitHub Pages** at https://mehty.github.io/.

## Tech Stack

- **Static Site Generator:** Jekyll 3.9.5 (via `github-pages` gem)
- **Theme:** Minima
- **Markdown:** Kramdown with GitHub-Flavored Markdown
- **CSS Framework:** W3.CSS (CDN), Bootstrap 3.3.7 (CDN)
- **Icons:** Font Awesome 4.7.0 (CDN)
- **Fonts:** Montserrat (Google Fonts)
- **Diagrams:** jekyll-mermaid plugin
- **No JavaScript framework** — vanilla JS only, no npm/Node tooling

## Repository Structure

```
mehty.github.io/
├── .github/
│   └── workflows/
│       └── jekyll-docker.yml   # CI/CD: builds site on push to master
├── _blogs/                     # Blog post source files (Markdown)
│   └── ai-ds-helper.md
├── blogs/
│   └── index.html              # Blog listing page (Liquid template)
├── images/                     # Images and assets used across the site
├── resume/                     # PDF resume files
├── _config.yml                 # Jekyll site configuration
├── Gemfile / Gemfile.lock      # Ruby dependency management
├── index.html                  # Main portfolio landing page
├── README.md
└── .gitignore                  # Ignores _site/ and .DS_Store
```

## Key Configuration

**`_config.yml`**
- `title: Mehty`
- `url: https://mehty.github.io`
- `markdown: kramdown`
- `theme: minima`
- `collections.blogs.output: true` with permalink `/blogs/:name/`
- `plugins: [jekyll-mermaid]`

**`.gitignore`**
- `_site/` — generated build output, never committed
- `.DS_Store` — macOS metadata

## Local Development

This project uses Ruby/Bundler. There is no Node.js or npm involved.

```bash
# Install dependencies
bundle install

# Serve locally with live reload
bundle exec jekyll serve

# Build for production (same as CI)
bundle exec jekyll build --future
```

The local server runs at `http://localhost:4000` by default. The `--future` flag is used in CI to include posts with future dates.

## Adding Blog Posts

1. Create a new Markdown file in `_blogs/` with this front matter:

```yaml
---
title: "Your Post Title"
description: "Brief description"
date: YYYY-MM-DD
layout: post
permalink: /blogs/your-post-slug/
---
```

2. Write content in Markdown below the front matter.
3. The post will automatically appear in `blogs/index.html` via the Liquid loop over `site.blogs`.

Mermaid diagrams are supported via the `jekyll-mermaid` plugin:

````mermaid
graph TD;
  A-->B;
````

## Deployment / CI

**Workflow:** `.github/workflows/jekyll-docker.yml`
- **Trigger:** Push to `master` branch or pull requests targeting `master`
- **Mechanism:** Docker container (`jekyll/builder:latest`)
- **Build command:** `jekyll build --future`
- **Output:** `_site/` directory (artifact, not committed)

GitHub Pages serves the site directly from the `master` branch. The Actions workflow validates the build; actual serving is handled by GitHub Pages infrastructure.

## Editing the Landing Page

`index.html` is a single-file portfolio page. It is **not** generated from a Jekyll layout — it is plain HTML with W3.CSS classes. Key sections (by anchor):

| Anchor | Content |
|--------|---------|
| `#home` | Hero / name banner |
| `#about` | Bio and featured projects |
| `#blogs` | Link to /blogs/ |
| `#music` | Embedded SoundCloud iframes |
| `#contact` | Location, phone, email |

Navigation is a fixed W3.CSS sidebar. The color scheme is black (`#1b1b1b`) with grey text.

## Conventions

- **No build step for CSS/JS** — all styling is done via CDN-linked frameworks and inline styles in HTML.
- **Images** go in `images/`. Use relative paths (e.g., `images/my_icon.jpg`).
- **Resumes** (PDFs) go in `resume/`.
- **Blog posts** go in `_blogs/` as `.md` files.
- Keep `_site/` out of version control — it is in `.gitignore`.
- The site is responsive via W3.CSS responsive classes (`w3-hide-small`, etc.).

## Social / Contact Links

Update these in `index.html` footer if they change:
- GitHub: https://github.com/mehty
- LinkedIn: https://www.linkedin.com/in/amirmehdi-sh
- Twitter: https://twitter.com/ASharifzad
- SoundCloud: https://soundcloud.com/amsharifzad
