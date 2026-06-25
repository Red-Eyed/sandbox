# Design Doc Section Templates

Copy-paste starting points for each section. Adapt to your project.

---

## Metadata

```markdown
**Author**: Your Name <you@example.com>
**Status**: Draft | Ready for Review | Approved
**Created**: YYYY-MM-DD
**URL**: http://go/your-project-design
```

---

## Objective

```markdown
## Objective

[One sentence. What does this project do and why does it matter to a stakeholder?]

Example: Improve application performance by adding a caching layer between the Trogdor web server and the Postgres database.
```

---

## Background

```markdown
## Background

[Why is the team taking on this project? What problem does it solve? Were there previous attempts?]

[Include enough context that a reader unfamiliar with your team can understand the doc without a verbal briefing.]
```

---

## Goals

```markdown
## Goals

- [Outcome in terms of user/team/company impact]
- [Another outcome]

❌ BAD: "Add Kubernetes to our infrastructure."
✅ GOOD: "Minimize outages related to deploying new app versions."
```

---

## Non-Goals

```markdown
## Non-Goals

- [Thing readers might assume is in scope but isn't]
  - [Brief explanation of why it's excluded or deferred]
- [Another non-goal]
```

---

## Scenarios

```markdown
## Scenarios

### Scenario: [Name]

1. [Actor] does [action].
2. System responds with [response].
3. [Actor] sees [result].
```

---

## Interfaces

```markdown
## Interfaces

### API

[Describe endpoints, request/response shapes, auth mechanism]

### CLI

[Describe commands, flags, exit codes]

### UI

[Rough sketches or descriptions — not pixel-perfect mockups]
```

---

## Dependencies / Infrastructure

```markdown
## Dependencies

- **Language**: [Go / Python / Rust / etc.] — [one-line justification]
- **Storage**: [Postgres / Redis / S3 / etc.] — [one-line justification]
- **Third-party packages**:
  - [package name]: [what it does and why chosen]
- **Infrastructure**: [Kubernetes / bare metal / serverless / etc.]
```

---

## SLOs

```markdown
## Service Level Objectives

| Metric | Target |
|---|---|
| User-facing HTTP p50 latency | ≤ 200ms |
| User-facing HTTP p99 latency | ≤ 2s |
| Availability | ≥ 99.9% |
| Throughput | ≥ 1000 req/s |
```

---

## Monitoring / Alerting

```markdown
## Monitoring

The following events page the on-call engineer:

- [Metric] exceeds [threshold] for [window]: [action]
- [Another alert condition]

Dashboards: [link]
```

---

## Security

```markdown
## Security

**Attack surface**: [Where does this system process potentially malicious data?]

**Trust boundaries**: [Where does data cross from lower to higher privilege?]

**Threats considered**:
- [Threat 1]: [Mitigation or acceptance reasoning]
- [Threat 2]: [Mitigation or acceptance reasoning]
```

---

## Privacy

```markdown
## Privacy

**Sensitive data handled**: [What PII or sensitive data does this system touch?]

**Retention**: [How long is data kept?]

**Access controls**: [Who can access it? Under what conditions?]

**Encryption**: [At rest: yes/no/details. In transit: yes/no/details.]
```

---

## Open Issues

```markdown
## Open Issues

### [Issue Title]

**Problem**: [What's unresolved?]

**Options**:
1. [Option A] — [tradeoffs]
2. [Option B] — [tradeoffs]

**Proposed solution**: [Your current lean, if any]

**Next step**: [Who needs to weigh in? What information is needed?]
```

---

## Resolved Issues

```markdown
## Resolved Issues

### [Issue Title]

**Decision**: [What was decided and why]

[Original open issue content preserved below for posterity]

[...]
```

---

## Alternatives Considered

```markdown
## Alternatives Considered

- **[Alternative A]**: [Why it was rejected — one or two sentences]
- **[Alternative B]**: [Why it was rejected — one or two sentences]
```

---

## Timeline

```markdown
## Timeline

| Milestone | Date | Description |
|---|---|---|
| M1 | YYYY-MM-DD | [Artifact: what stakeholders can see/use] |
| M2 | YYYY-MM-DD | [Next artifact] |
| Launch | YYYY-MM-DD | [Production rollout] |
```

---

## Glossary

```markdown
## Glossary

- **[Term]**: [Definition. Link to internal docs if available.]
- **[Acronym]**: [What it stands for + brief explanation.]
```
