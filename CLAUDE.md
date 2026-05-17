# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AstroPaper — an Astro 5 blog theme (TypeScript, Tailwind v4, Pagefind search). Blog content is markdown in `src/data/blog/`.

## Package manager

Use **npm** (not pnpm), even though a `pnpm-lock.yaml` and `pnpm-workspace.yaml` exist in the tree. `package-lock.json` is the source of truth.

## Commands

- `npm run dev` — start Astro dev server
- `npm run build` — runs `astro check` (type-checks `.astro` + content), `astro build`, then builds the Pagefind search index and copies it into `public/pagefind/`. The Pagefind copy step means a build must complete before search works in `dev` against built output.
- `npm run preview` — preview built site
- `npm run sync` — regenerate `astro:content` types after editing `content.config.ts` or adding new frontmatter fields
- `npm run lint` — ESLint (flat config in `eslint.config.js`, includes `eslint-plugin-astro` + `typescript-eslint`)
- `npm run format` / `npm run format:check` — Prettier with `prettier-plugin-astro` and `prettier-plugin-tailwindcss`

No test runner is configured.

## Architecture

**Content pipeline.** Blog posts are a single Astro content collection defined in `src/content.config.ts`, loaded via `glob` from `src/data/blog/**/[^_]*.md` (files prefixed with `_` are ignored). The Zod schema there is the contract for post frontmatter — adding/removing fields requires updating consumers in `src/pages/` and `src/utils/` and running `npm run sync`.

**Routing.** `src/pages/` is file-based: `posts/`, `tags/`, `archives/` produce post lists, tag pages, and archive views. Dynamic OG images are generated per-post via `src/pages/og.png.ts` using `satori` + `@resvg/resvg-js` with templates in `src/utils/og-templates/`. RSS (`rss.xml.ts`), `robots.txt.ts`, and a static `404.astro` round out the routes.

**Site config.** `src/config.ts` exports the `SITE` object (URL, author, pagination, locale, `showArchives`, etc.) — this drives `astro.config.ts` (sitemap filter, `site` URL) and most layouts. Change site-wide behavior here, not in individual pages.

**Markdown rendering.** Configured in `astro.config.ts`:
- `remark-toc` + `remark-collapse` (collapses a heading named "Table of contents")
- Shiki with dual themes (`min-light` / `night-owl`) and custom transformers, including a local `transformerFileName` in `src/utils/transformers/`
- Notation transformers from `@shikijs/transformers` enable `// [!code highlight]`, `// [!code ++]`, etc. in code fences.

**Styling.** Tailwind v4 via `@tailwindcss/vite` (no `tailwind.config.*`); customize tokens in `src/styles/`. Light/dark mode is built-in.

**Search.** Pagefind indexes the built `dist/` output at build time; the generated index is copied into `public/pagefind/` so the static search UI on `src/pages/search.astro` can fetch it.

**Path alias.** `@/*` → `src/*` (see `tsconfig.json`); use it instead of long relative paths.

## Conventions

- Conventional Commits (see `cz.yaml`); commitizen-friendly.
- Post utilities in `src/utils/` (`getSortedPosts`, `postFilter`, `getUniqueTags`, `getPostsByTag`, `getPostsByGroupCondition`) centralize filtering of drafts/future posts — reuse them rather than re-filtering collection entries in pages.
