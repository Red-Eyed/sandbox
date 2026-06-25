---
name: design-doc
description: >
  Use this skill whenever the user wants to write, draft, review, or improve a software design document (also called a design doc, tech spec, or RFC). Trigger when the user mentions "design doc", "tech spec", "RFC", "architecture doc", "system design document", or asks to "write up the design", "document the architecture", or "create a spec" for a feature, system, or project. Also trigger when the user wants to review an existing design doc for completeness or quality, or when they ask what sections a design doc should have.
---

# Software Design Document Skill

Based on battle-tested practices from Google and Microsoft (source: [Refactoring English](https://refactoringenglish.com/excerpts/write-an-effective-design-doc/)).

## Core Philosophy

A design doc should articulate the **hard problems** you're solving and help teammates give you useful feedback. It is NOT a specification of every implementation detail — that defeats the purpose.

**The key filter: "What's the cost of getting it wrong?"**
- High reversal cost → document it (language choice, storage backend, auth model, API shape)
- Low reversal cost → skip it (pagination style, button label, sort order)

## When to Write a Design Doc

Write one if you answer "yes" to any of these:
- Will multiple people coordinate to implement this?
- Will it take more than 3 months of full-time work?
- Will it run in production for several years?
- Does it involve cross-team collaboration?
- Are goals/requirements ambiguous?
- Are there catastrophic risks preventable at design time (security flaws, legal risks)?

Two or more "yes" answers → almost certainly worth writing.

## How Much to Invest

Calibrate to the project. A one-pager is fine for small work; a 50-page document requiring multi-team signoff is right for large distributed systems. Sometimes the right investment is zero.

---

## Standard Sections

Pick the subset appropriate for your project. Not every section belongs in every doc.

### Required Always
| Section | What to put there |
|---|---|
| **Title** | Short, distinctive, evocative. Easy to say aloud. Not "Project Flying Silver Horse". |
| **Metadata** | Author (name + email), date created, status (Draft / Ready for Review / Approved), authoritative URL |
| **Objective** | One sentence. Plain language any stakeholder understands. |
| **Background** | Why are we doing this? What problem does it solve? Prior attempts? |
| **Goals** | High-level outcomes in terms of user/team/company impact. NOT implementation details. |
| **Non-goals** | Explicitly name things readers might wrongly assume are in scope. |

### Recommended for Most Projects
| Section | What to put there |
|---|---|
| **Alternatives Considered** | Briefly: what you rejected and why. Preempts "why didn't you do X?" |
| **Interfaces** | API shapes, CLI semantics, file formats, or rough UI sketches (not pixel-perfect). |
| **Dependencies / Infrastructure** | Language, hardware, storage. Focus on what's hard to change later. |
| **Open Issues** | Unresolved questions with: problem statement, options, next step. |
| **Resolved Issues** | Moved from Open Issues after decision; retain the original discussion. |
| **Timeline** | Milestones with dates. Prefer milestones that create artifacts for stakeholders. |

### Include When Relevant
| Section | What to put there |
|---|---|
| **Scenarios** | Walk-throughs of how the completed system works in the real world. |
| **Diagrams** | Data flow, component relationships, communication protocols. Use editable tools (Excalidraw, draw.io, Mermaid, D2). Store source, not just images. |
| **Related Documents** | Links to test plans, functional specs, related design docs. |
| **Glossary** | Internal tool names, team-specific jargon, or acronyms that external readers won't know. |
| **Constraints** | Budget, client requirements, infrastructure limits that shape design choices. |
| **SLOs** | Concrete, measurable performance targets (uptime %, latency p50/p95, throughput). |
| **Monitoring / Alerting** | How you'll detect SLO failures. What events page on-call. |
| **Security** | Threats considered, attack surface, trust boundaries. Document even if risk is low. |
| **Privacy** | Sensitive data handled, retention policy, access controls, encryption at rest/in transit. |
| **Legal** | Regulated domains, license choices, contractual constraints. |
| **Logging** | What events are logged, log levels, retention, access control, sensitive data exclusions. |

---

## Writing Process

1. **Write the Background and Goals first** — if you can't state these clearly, you're not ready to design.
2. **Document hard decisions with rationale** — especially language, storage, auth, API shape.
3. **Capture open issues as you go** — don't let unknowns block the doc; surface them explicitly.
4. **Diagrams before prose** for architecture — reviewers can't see your mental model.
5. **Background section must be self-contained** — some readers won't hear a verbal intro before reading.

## Review Process Tips

- Share early (even draft) to surface disagreements before you're invested.
- Assign explicit reviewers with deadlines.
- State what kind of feedback you want ("is the approach sound?" vs. "line edits").
- Move resolved issues to the Resolved section — show that you acted on feedback.

---

## Section Templates

See `references/templates.md` for copy-paste templates for each section with example content.

---

## Output Format

When drafting a design doc, output a Markdown file (`.md`). Use `##` for top-level sections and `###` for subsections. Include a table of contents at the top if the doc has more than ~6 sections.

Never output as `.docx` unless the user explicitly asks.
