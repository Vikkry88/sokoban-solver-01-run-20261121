# Push-optimal Sokoban solver

[![make.fun competition](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Vikkry88/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Vikkry88/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

An entry for [sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121)
on [make.fun](https://make.fun). `main.ts` (Node 22, no dependencies) reads
every level under `/input/<id>/level.txt` and writes the player's moves to
`/output/<id>/solution.txt`.

## Approach

Every generated level is a one-wide hallway with independent rooms off it,
and a box pushed into the hallway can never be pushed back, so a level's
push optimum is the sum of its rooms' optima. The solver therefore:

- splits each level into rooms by cutting the hallway row out of the floor;
- solves every room with A* over pushes (not moves), where a state is the
  sorted box squares plus the player's reachable region named by its
  lowest square;
- guides the search with the cheapest box-to-goal assignment of
  single-box push distances (each box pulled back from every goal over
  open floor), an admissible heuristic, so the first solution settled is
  push-optimal;
- prunes pushes onto dead squares (no single box reaches a goal from
  there) and into frozen 2x2 blocks of walls and boxes;
- retries a room with a heavier heuristic weight if its expansion budget
  runs out, so it still answers rather than giving up.

All budgets count expansions, never the clock, and every tie is broken by
a fixed order, so two runs write the same bytes. On the kit's practice set
it solves all five levels in their optimal push counts (13, 16, 29, 47
and 60 pushes) in well under a second.

## License

MIT, see `LICENSE`.
