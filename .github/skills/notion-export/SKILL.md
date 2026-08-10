---
name: notion-export
description: Analyze experiment results from notebooks, figures, and tables, then export a structured report to Notion using the template Objective, Setup, Observations, and Follow-ups.
---

# Notion Result Export Skill

Use this skill when the user asks to summarize analytical results from notebooks, plots, or tabular outputs and publish them to Notion.

## When To Use

- User asks to analyze a notebook and export to Notion.
- User provides figures/screenshots and wants findings documented.
- User asks for a structured experiment report with explicit sections.

## Inputs You Should Gather

- Source artifacts:
  - Notebook path(s), figure files, table files, or pasted outputs.
- Analysis context:
  - Dataset name, task type, models, metrics, and split setup.
- Notion destination:
  - Target page URL/ID or database URL/ID.

## Default Notion Destination

Use this database by default when the user does not provide a destination:

- database ID: 2e5cf63bea3880c39bb6d1e048c69397

ID handling rule:

- Treat `2e5cf63bea3880c39bb6d1e048c69397` as a **database ID**.
- Do **not** call data-source retrieval with this ID directly.
- First retrieve the database (`retrieve-a-database`), then get its child `data_sources[0].id`.
- Use the resolved data source ID for page creation under the database.

If export to the default destination fails due to permissions, ask the user to share it with the active integration and retry.

## Required Report Template

Always export content using these section headings in this exact order:

1. Objective
2. Setup
3. Observations
4. Follow-ups

## Workflow

1. Inspect sources
   - Notebook: read markdown/code cells and persisted outputs.
   - Figures: capture trends, outliers, and comparative performance.
   - Tables: extract key metrics, best/worst rows, and deltas.
2. Validate evidence
   - Base claims on visible outputs only.
   - Mark assumptions explicitly when evidence is incomplete.
3. Draft concise report
   - Objective: what question the experiment answered.
   - Setup: data, model variants, metrics, and evaluation protocol.
   - Observations: numbered findings with metric-oriented evidence.
   - Follow-ups: concrete next experiments or checks.
4. Export to Notion
   - Create a page under the provided parent (page or database).
  - If the destination is a database ID, resolve it to a data source ID first (database -> data_sources[0].id).
  - If the destination is already a data source ID, use it directly.
  - Set the report title as the Notion page title property (for database rows, set the Name/title property).
   - Preserve the exact template headings.
   - Prefer short bullets and numbered findings.

## Quality Bar

- Be specific about metric direction (higher/lower is better).
- Include notable failure modes and outliers.
- Distinguish facts from interpretation.
- Keep it concise and decision-oriented.

## Analysis Result Formatting Requirements

- Format pseudo code, variable-value snippets, and shape-like outputs using inline Markdown code syntax (for example, `shape=(1,10631,82,192)`).
- Bold important sentences and critical numbers so key outcomes are visually prominent.

## Export Block Structure (Notion)

Create Notion blocks in this order:

1. Paragraph: source references
2. Heading 2: Objective
3. Paragraph or bullets
4. Heading 2: Setup
5. Bullets
6. Heading 2: Observations
7. Numbered list
8. Heading 2: Follow-ups
9. TODO list

## Failure Handling

- If Notion returns object_not_found or permission errors:
  - Verify object type and endpoint pairing before assuming permissions:
    - Database ID -> `retrieve-a-database`
    - Data source ID -> `retrieve-a-data-source`
  - Ask the user to share the page/database with the active integration.
  - Retry with both database ID and view ID candidates from URL.
- If notebook outputs are missing:
  - Report that findings are based on code intent only and ask whether to run cells.
