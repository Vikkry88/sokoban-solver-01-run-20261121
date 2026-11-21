// Room-decomposed push-optimal Sokoban solver.
//
// Each generated level is a one-wide hallway with independent rooms off it,
// and a box pushed into the hallway can never come back, so the level's
// push optimum is the sum of its rooms' optima. This entry therefore
// splits a level into rooms and solves every room on its own with A*
// over pushes (not moves):
//
//   - a state is the sorted box squares plus the player's reachable
//     region, named by its lowest square, so states that differ only by
//     where the player stands inside one region merge;
//   - the heuristic is the cheapest box-to-goal assignment of single-box
//     push distances (each box alone, pulled back from every goal over
//     open floor), which never overestimates the pushes still needed;
//   - a push onto a square no single box can reach a goal from, or into
//     a frozen 2x2 block of walls and boxes with a box off its goal, is
//     pruned: neither can occur in a solution.
//
// With weight 1 the first solved state A* settles is push-optimal. If a
// room's work budget runs out, the room is retried with a heavier
// heuristic weight, which still searches by pushes but settles for a
// near-optimal answer rather than none. All budgets count expansions,
// never the clock, and every tie is broken by a fixed order, so two runs
// write the same bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, in A* expansions (deterministic; no clock is ever read).
const ROOM_BUDGET = 900_000;
const RETRY_BUDGET = 900_000;
const WHOLE_BUDGET = 900_000;

const INF = 1_000_000;

// Direction tables in a fixed order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP_LETTER = ['u', 'd', 'l', 'r'];
const PUSH_LETTER = ['U', 'D', 'L', 'R'];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // The outer frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Single-box push distances to each goal: dists[g][cell] is the fewest
// pushes that bring one box from cell onto goal g, ignoring other boxes.
// Pulled backwards: a box at x came from x-d only when the player could
// stand at x-2d. INF where no such push sequence exists (a dead square).
function pushDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  return goals.map((goal) => {
    const dist = new Int32Array(size).fill(INF);
    const queue: number[] = [goal];
    dist[goal] = 0;
    for (let head = 0; head < queue.length; head++) {
      const x = queue[head]!;
      for (let dir = 0; dir < 4; dir++) {
        const d = DY[dir]! * w + DX[dir]!;
        const from = x - d;
        const stand = from - d;
        if (from < 0 || from >= size || stand < 0 || stand >= size) continue;
        if (wall[from] || wall[stand] || dist[from] !== INF) continue;
        dist[from] = dist[x]! + 1;
        queue.push(from);
      }
    }
    return dist;
  });
}

// All permutations of [0..n), generated once per box count.
const permCache = new Map<number, number[][]>();
function permutations(n: number): number[][] {
  const hit = permCache.get(n);
  if (hit) return hit;
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const build = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      build();
      cur.pop();
      used[i] = false;
    }
  };
  build();
  permCache.set(n, out);
  return out;
}

// Cheapest assignment of boxes to distinct goals under single-box push
// distances. Admissible: every real solution pushes each box at least
// its single-box distance, and distinct boxes end on distinct goals.
function assignmentCost(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 7) {
    // Too many boxes to permute: fall back to each box's nearest goal.
    let sum = 0;
    for (const b of boxes) {
      let best = INF;
      for (const dist of dists) if (dist[b]! < best) best = dist[b]!;
      sum += best;
      if (sum >= INF) return INF;
    }
    return sum;
  }
  let best = INF;
  for (const perm of permutations(n)) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += dists[perm[i]!]![boxes[i]!]!;
      if (sum >= best) break;
    }
    if (sum < best) best = sum;
  }
  return best;
}

// Marks every square the player can reach from `from` with `occupied`
// boxes in the way, stamping `seen` with `stamp`; returns the lowest
// reachable square, the region's name.
function floodRegion(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let top = 0;
  stack[top++] = from;
  seen[from] = stamp;
  while (top > 0) {
    const p = stack[--top]!;
    if (p < low) low = p;
    for (let dir = 0; dir < 4; dir++) {
      const q = p + DY[dir]! * w + DX[dir]!;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[top++] = q;
      }
    }
  }
  return low;
}

// Whether the box just pushed to `t` completes a frozen 2x2 block of
// walls and boxes that holds a box off its goal: no box in such a block
// can ever move again, so the push is dead.
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes, in lower
// LURD letters; direction order breaks ties, so the walk is fixed.
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let head = 0; head < queue.length; head++) {
    const p = queue[head]!;
    if (p === to) break;
    for (let dir = 0; dir < 4; dir++) {
      const q = p + DY[dir]! * w + DX[dir]!;
      if (q < 0 || q >= wall.length || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP_LETTER[dir]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

type PushSolution = { pushes: number; moves: string; expansions: number };

// A* over pushes for one board (a room, or a whole level as fallback).
// `weight` scales the heuristic: 1 is push-optimal; larger values trade
// optimality for speed. Returns null when the expansion budget runs out
// or no solution exists. Fully deterministic.
function solvePushes(
  w: number,
  wall: Uint8Array,
  goals: number[],
  startBoxes: number[],
  startPlayer: number,
  budget: number,
  weight: number,
): PushSolution | null {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dists = pushDistances(w, wall, goals);
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);
  if (assignmentCost(startBoxes, dists) >= INF) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const numericKey = startBoxes.length <= 5;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKey) return `${low}:${boxes.join(',')}`;
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  // State store, in discovery order.
  const stateBoxes: number[][] = [startBoxes.slice()];
  const statePlayer: number[] = [startPlayer];
  const stateG: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];

  for (const b of startBoxes) occupied[b] = 1;
  const startLow = floodRegion(w, wall, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  const bestG = new Map<number | string, number>([[keyOf(startBoxes, startLow), 0]]);

  // Binary heap on (f, then deeper g first, then discovery order).
  type Entry = [number, number, number, number]; // f, -g, seq, state
  const heap: Entry[] = [];
  let seq = 0;
  const less = (a: Entry, b: Entry): boolean => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: Entry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): Entry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  heapPush([weight * assignmentCost(startBoxes, dists), 0, seq++, 0]);
  let expansions = 0;
  let found = -1;

  while (heap.length > 0) {
    const [, negG, , s] = heapPop();
    const g = -negG;
    if (stateG[s] !== g) continue;
    const boxes = stateBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expansions >= budget) return null;
    expansions++;

    for (const b of boxes) occupied[b] = 1;
    floodRegion(w, wall, occupied, statePlayer[s]!, seen, ++stamp, stack);
    const here = stamp;

    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let dir = 0; dir < 4; dir++) {
        const d = DY[dir]! * w + DX[dir]!;
        const target = b + d;
        const stand = b - d;
        if (target < 0 || target >= size || stand < 0 || stand >= size) continue;
        if (seen[stand] !== here || wall[target] || occupied[target]) continue;
        // A box on a square that reaches no goal can never be solved.
        let reachable = false;
        for (const dist of dists) if (dist[target]! < INF) { reachable = true; break; }
        if (!reachable) continue;

        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, target);
        let next: number[] | null = null;
        let key: number | string = 0;
        let h = INF;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((x, y) => x - y);
          h = assignmentCost(next, dists);
          if (h < INF) key = keyOf(next, floodRegion(w, wall, occupied, b, mark, ++stamp, stack));
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (frozen || next === null || h >= INF) continue;

        const ng = g + 1;
        const seenG = bestG.get(key);
        if (seenG !== undefined && seenG <= ng) continue;
        bestG.set(key, ng);
        const idx = stateBoxes.length;
        stateBoxes.push(next);
        statePlayer.push(b);
        stateG.push(ng);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(dir);
        heapPush([ng + weight * h, -ng, seq++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }

  if (found < 0) return null;

  // Replay the push path, walking the player to each push square.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes.slice();
  for (const s of path) {
    const from = pushFrom[s]!;
    const dir = pushDir[s]!;
    const d = DY[dir]! * w + DX[dir]!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH_LETTER[dir]!;
    boxes = stateBoxes[s]!;
    player = from;
  }
  return { pushes: path.length, moves, expansions };
}

// The level's rooms: cut the hallway row (the player's row) out of the
// floor and take each remaining part that holds boxes. Cutting only the
// hallway squares a box can never leave (dead squares) is tried first;
// cutting the whole row is the fallback. Null when some part holds more
// or fewer boxes than goals, or a box starts in the hallway.
function roomsOf(level: Level): { boxes: number[]; goals: number[] }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const dists = pushDistances(w, wall, goals);
  const live = (cell: number): boolean => dists.some((dist) => dist[cell]! < INF);
  const row = Math.floor(player / w);

  const attempt = (cut: Uint8Array): { boxes: number[]; goals: number[] }[] | null => {
    if (boxes.some((b) => cut[b] === 1)) return null;
    const part = new Int32Array(wall.length).fill(-1);
    const rooms: { boxes: number[]; goals: number[] }[] = [];
    for (let s = 0; s < wall.length; s++) {
      if (wall[s] || cut[s] || part[s] >= 0) continue;
      const squares = [s];
      part[s] = rooms.length;
      for (let i = 0; i < squares.length; i++) {
        for (let dir = 0; dir < 4; dir++) {
          const q = squares[i]! + DY[dir]! * w + DX[dir]!;
          if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
          part[q] = rooms.length;
          squares.push(q);
        }
      }
      rooms.push({ boxes: [], goals: [] });
    }
    for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
    for (const g of goals) {
      if (part[g]! < 0) return null;
      rooms[part[g]!]!.goals.push(g);
    }
    if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
    return rooms.filter((r) => r.boxes.length > 0);
  };

  const deadCut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live(i)) deadCut[i] = 1;
  const byDead = attempt(deadCut);
  if (byDead !== null) return byDead;

  const rowCut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i]) rowCut[i] = 1;
  return attempt(rowCut);
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.player < 0 || level.boxes.length === 0) return '';
  const rooms = roomsOf(level);

  if (rooms !== null && rooms.length > 0) {
    let player = level.player;
    let moves = '';
    const done = new Array<boolean>(rooms.length).fill(false);
    let ok = true;
    for (let r = 0; r < rooms.length; r++) {
      const room = rooms[r]!;
      // Other rooms' boxes stand as walls: on their goals once solved,
      // where they started until then. Boxes never leave their room,
      // so this changes nothing the room's own search can reach.
      const wall = level.wall.slice();
      for (let o = 0; o < rooms.length; o++) {
        if (o === r) continue;
        for (const b of done[o] ? rooms[o]!.goals : rooms[o]!.boxes) wall[b] = 1;
      }
      const roomBoxes = room.boxes.slice().sort((a, b) => a - b);
      const roomGoals = room.goals.slice().sort((a, b) => a - b);
      let solved = solvePushes(level.w, wall, roomGoals, roomBoxes, player, ROOM_BUDGET, 1);
      if (solved === null) solved = solvePushes(level.w, wall, roomGoals, roomBoxes, player, RETRY_BUDGET, 3);
      if (solved === null) solved = solvePushes(level.w, wall, roomGoals, roomBoxes, player, RETRY_BUDGET, 10);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      done[r] = true;
    }
    if (ok) return moves;
  }

  // Fallback: search the whole level at once.
  let solved = solvePushes(level.w, level.wall, level.goals, level.boxes, level.player, WHOLE_BUDGET, 1);
  if (solved === null) solved = solvePushes(level.w, level.wall, level.goals, level.boxes, level.player, WHOLE_BUDGET, 3);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
