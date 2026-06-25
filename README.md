# Claude skills

Version-controlled Claude Code skills, kept on the **orphan `claude-skills` branch**
of the `sandbox` repo. This branch has no shared history with `main` — it exists only
to track skills independently of the project code.

## Layout

```
<skill-name>/
  SKILL.md                 # required: frontmatter `name:` must equal the directory name
  references/              # optional supporting material
```

Each top-level directory is one skill.

## How these become global skills

This directory is a **git worktree** of the `claude-skills` branch, checked out at a
stable path that survives branch switches in the main repo:

```
git worktree add --orphan -b claude-skills ~/.claude/skills-repo
```

Claude Code only discovers skills that are top-level directories under
`~/.claude/skills/`, so each skill is exposed with a symlink:

```
~/.claude/skills/<skill-name>  ->  ~/.claude/skills-repo/<skill-name>
```

Because the symlink targets the worktree (not a specific checkout of the main repo),
the skill stays available no matter which branch the main repo is on.

## Adding a new skill

1. Create `~/.claude/skills-repo/<skill-name>/SKILL.md` (and any `references/`).
2. Commit it on the `claude-skills` branch (run git from inside this worktree).
3. Link it globally:

   ```
   ln -s ~/.claude/skills-repo/<skill-name> ~/.claude/skills/<skill-name>
   ```

## Current skills

- `design-doc` — write, review, and improve software design docs / tech specs / RFCs.
