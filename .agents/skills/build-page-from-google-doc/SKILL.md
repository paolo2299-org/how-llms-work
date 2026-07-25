---
name: build-page-from-google-doc
description: Create a new How LLMs Work site page or completely replace an existing page from copy supplied in a Google Doc. Use when the user provides a Google Docs URL and asks to create, import, rewrite, or replace any educational page, including a homepage, overview, index, landing, or in-depth page, while preserving their copy, completing bracketed editorial placeholders, following this repository's page conventions, and separately flagging correctness, grammar, or style concerns.
---

# Build Page from Google Doc

Turn a supplied Google Doc into a finished site page in this repository. Preserve the author's wording while supplying the implementation, explanatory visuals, and content explicitly requested by placeholders.

## Gather the source and target

1. Read `AGENTS.md` and inspect the current repository before editing. Treat its current structure, comparable pages, shared styles, route definitions, navigation, and page tests as the source of truth.
2. Read the complete Google Doc using a connected Google Drive tool when available. Otherwise open a publicly shared URL. Capture all body content and meaningful structure, including headings, lists, tables, code, links, captions, and footnotes.
3. If the document cannot be accessed, stop and ask the user to make it readable, connect Google Drive, paste the copy, or provide an export. Do not reconstruct the document from search snippets or memory.
4. Determine the target and its role from the request and repository:
   - Replace the page when its route or template already exists.
   - Create a page when no corresponding page exists.
   - Classify it as a high-level page, such as a homepage, overview, index, or landing page, or an in-depth page within the site's hierarchy.
   - Ask only when multiple plausible targets remain and choosing one would materially alter the site.

## Protect the author's copy

1. Preserve the supplied copy exactly, including spelling, punctuation, capitalisation, terminology, and paragraph order. Do not silently proofread or rewrite it.
2. Make only these textual changes without separate approval:
   - Replace text that clearly functions as a bracketed editorial placeholder with original content that satisfies it.
   - Make mechanical HTML adaptations that do not alter visible wording, such as escaping characters or attaching link targets.
3. Do not treat brackets in code, mathematical notation, citations, links, or ordinary prose as placeholders merely because they use square brackets.
4. Record each substantive piece of assistant-authored placeholder content so it can be identified in the final response.
5. Note potential factual, grammatical, or stylistic issues while reading, but keep the supplied wording on the page. If an issue would make implementation unsafe, impossible, or materially misleading in an interactive element, explain the conflict and ask before proceeding.

## Implement the page

1. Use semantic, accessible HTML and the repository's existing visual language. Prefer maintainable HTML, CSS, SVG, and vanilla JavaScript over new dependencies or a frontend build step.
2. For a replacement, replace the page-specific content completely while retaining shared site shell elements, the established route, and useful navigation unless the document or user explicitly directs otherwise. Remove obsolete page-only markup, styles, scripts, and tests.
3. For a new page:
   - Add the template and route using the conventions in `app/templates/` and `app/routes/pages.py`.
   - Place it appropriately in the site's navigation hierarchy. Give high-level pages suitable top-level or section navigation; give in-depth pages related, section, or previous/next navigation when existing patterns use it.
   - Use a short, stable, kebab-case route that matches the page topic.
4. Match the layout and navigational prominence to the page's role. Do not force an in-depth-page structure or previous/next sequence onto a high-level page. Use page role to guide implementation, not to rewrite the source copy.
5. Translate document formatting faithfully. Build any diagrams, examples, captions, tables, or interactions requested by placeholders in the style of comparable educational pages. Keep them responsive and usable by keyboard and screen readers where interactive.
6. Keep changes scoped to the requested page and the routing, navigation, styles, scripts, and tests it genuinely requires.

## Verify the result

1. Compare the finished visible text with the Google Doc in order. Confirm that every source paragraph appears unchanged except for completed placeholders and unavoidable non-visible HTML mechanics.
2. Check headings, code, links, tables, footnotes, diagrams, navigation, responsive layout, and accessibility labels.
3. Add or update focused tests in `tests/test_pages.py` for the route, distinctive page content, and navigation. Follow the repository's existing Flask test patterns.
4. Run the focused page tests, then the broader repository test command specified by `AGENTS.md` when practical. Perform browser-based visual verification when the available environment supports it.
5. Do not claim a check passed unless it was actually run.

## Report back

Lead with the completed page and summarize:

- Files changed and verification performed.
- Placeholder content added, described briefly.
- Potential issues in the supplied copy under separate `Correctness`, `Grammar`, and `Style` labels, quoting only the minimum excerpt needed to locate each issue.

State explicitly when a category has no issues. Present suggestions only; do not imply that they were silently applied.
