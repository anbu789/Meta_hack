---
title: Pharmavigil
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# PharmaVigil — Pharmacovigilance Signal Detection Environment

An OpenEnv-compliant RL environment where an AI agent acts as a pharmacovigilance analyst.

## Run Locally
```bash
uvicorn app.main:app --reload --port 7860
```

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take action |
| GET | `/state` | Current state |
| GET | `/tasks` | List all tasks |
| POST | `/grader` | Get final score |

## Baseline Scores
| Task | Score |
|---|---|
| Task 1 — Classification | 0.300 |
| Task 2 — ROR Signal Detection | 0.350 |
| Task 3 — Masking Detection | 0.600 |

## openenv validate
✅ `[OK] Meta_hack: Ready for multi-mode deployment`