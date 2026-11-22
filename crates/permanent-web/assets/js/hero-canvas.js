// Calm blue-and-yellow alternating-cycle (worm) dimer field for the landing hero.
// Runs real closed alternating-cycle flips at a restrained cadence, pauses when
// scrolled out of view, and renders a static composition under reduced motion.

const PALETTES = {
  light: {
    bg: "#FAFBFD",
    wire: "rgba(0, 30, 95, 0.11)",
    rowNode: "#001E5F",
    colFill: "#FAFBFD",
    colStroke: "rgba(0, 30, 95, 0.45)",
    dimerH: "#001E5F", // URochester Navy
    dimerHSoft: "#0066FD", // Arpeggio Azure
    dimerV: "#D49000", // WCAG-friendly Rochester Gold on paper
    dimerVInner: "#FFD82B", // Dandelion Yellow core highlight
    cycleRing: "rgba(255, 216, 43, 0.55)",
    cycleOutline: "#0066FD",
  },
  dark: {
    bg: "#050C22",
    wire: "rgba(183, 211, 255, 0.12)",
    rowNode: "#B7D3FF",
    colFill: "#050C22",
    colStroke: "rgba(183, 211, 255, 0.42)",
    dimerH: "#66A2FF", // Connective Cornflower
    dimerHSoft: "#3B82F6",
    dimerV: "#FFD82B", // Dandelion Yellow
    dimerVInner: "#FFE95F",
    cycleRing: "rgba(255, 216, 43, 0.32)",
    cycleOutline: "#FFE95F",
  },
};

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

function createLattice(cols, rows) {
  // Ensure even total sites and even columns for brick covering.
  cols = Math.max(6, cols + (cols % 2));
  rows = Math.max(6, rows + (rows % 2));
  const total = cols * rows;
  // partner[idx] = neighbor idx in the current dimer covering.
  const partner = new Int32Array(total);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x += 2) {
      const a = y * cols + x;
      const b = a + 1;
      partner[a] = b;
      partner[b] = a;
    }
  }
  return { cols, rows, total, partner };
}

function neighborsOf(lattice, idx) {
  const { cols, rows } = lattice;
  const x = idx % cols;
  const y = (idx - x) / cols;
  const out = [];
  if (x > 0) out.push(idx - 1);
  if (x + 1 < cols) out.push(idx + 1);
  if (y > 0) out.push(idx - cols);
  if (y + 1 < rows) out.push(idx + cols);
  return out;
}

// Find a simple alternating cycle using a short loop-erased worm walk.
function findAlternatingCycle(lattice, startEven = null, maxLen = 12) {
  const { cols, rows, total, partner } = lattice;
  let u0 = startEven;
  if (u0 === null || u0 < 0 || u0 >= total) {
    // Pick a random even-sublattice site.
    const x = Math.floor(Math.random() * cols);
    const y = Math.floor(Math.random() * rows);
    u0 = y * cols + x;
    if ((x + y) % 2 !== 0) {
      u0 = y * cols + ((x + 1) % cols);
    }
  } else {
    const x = u0 % cols;
    const y = (u0 - x) / cols;
    if ((x + y) % 2 !== 0) u0 = partner[u0];
  }

  const walkU = [u0];
  const walkV = [partner[u0]];
  const posV = new Map([[partner[u0], 0]]);

  let curU = u0;
  for (let step = 0; step < 36; step++) {
    const curV = partner[curU];
    const nbrs = neighborsOf(lattice, curU).filter((v) => v !== curV);
    if (nbrs.length === 0) return null;
    const chosenV = nbrs[Math.floor(Math.random() * nbrs.length)];
    if (posV.has(chosenV)) {
      const start = posV.get(chosenV);
      const end = walkU.length - 1;
      const len = end - start + 1;
      if (len >= 2 && len <= maxLen) {
        const oldPairs = [];
        const newPairs = [];
        for (let i = start; i <= end; i++) {
          const u = walkU[i];
          const oldV = walkV[i];
          const newV = i === end ? chosenV : walkV[i + 1];
          oldPairs.push([u, oldV]);
          newPairs.push([u, newV]);
        }
        return { oldPairs, newPairs };
      }
      return null;
    }
    const nextU = partner[chosenV];
    posV.set(chosenV, walkV.length);
    walkU.push(nextU);
    walkV.push(chosenV);
    curU = nextU;
  }
  return null;
}

function applyCycle(lattice, cycle) {
  for (const [u, v] of cycle.newPairs) {
    lattice.partner[u] = v;
    lattice.partner[v] = u;
  }
}

class HeroScene {
  constructor(canvas, theme) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.theme = theme === "dark" ? "dark" : "light";
    this.reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.visible = true;
    this.disposed = false;
    this.rafId = null;
    this.timerId = null;
    this.activeAnim = null; // { oldPairs, newPairs, startMs, durationMs }
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.spacing = 44;
    this.offsetX = 0;
    this.offsetY = 0;
    this.lattice = null;

    this.onResize = this.resize.bind(this);
    this.onClick = this.handleClick.bind(this);
    this.onVisibility = () => {
      if (document.hidden) {
        this.pauseLoop();
      } else if (this.visible && !this.reducedMotion) {
        this.scheduleNextFlip();
      }
    };

    this.resizeObserver = new ResizeObserver(this.onResize);
    this.resizeObserver.observe(canvas);

    this.intersectionObserver = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        this.visible = Boolean(entry && entry.isIntersecting);
        if (this.visible && !this.reducedMotion) {
          this.draw(performance.now());
          this.scheduleNextFlip();
        } else {
          this.pauseLoop();
        }
      },
      { threshold: 0.05 },
    );
    this.intersectionObserver.observe(canvas);

    canvas.addEventListener("click", this.onClick);
    document.addEventListener("visibilitychange", this.onVisibility);

    this.resize();
    if (!this.reducedMotion) {
      this.scheduleNextFlip(900);
    }
  }

  resize() {
    if (this.disposed) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(320, Math.floor(rect.height));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w === this.width && h === this.height && dpr === this.dpr && this.lattice) {
      this.draw(performance.now());
      return;
    }
    this.width = w;
    this.height = h;
    this.dpr = dpr;
    this.canvas.width = Math.floor(w * dpr);
    this.canvas.height = Math.floor(h * dpr);

    this.spacing = w < 640 ? 38 : 48;
    const cols = Math.min(26, Math.max(8, Math.floor((w + this.spacing) / this.spacing)));
    const rows = Math.min(18, Math.max(8, Math.floor((h + this.spacing) / this.spacing)));
    this.lattice = createLattice(cols, rows);
    // Warm up with real alternating-cycle moves for an organic balanced mix.
    for (let i = 0; i < 140; i++) {
      const c = findAlternatingCycle(this.lattice, null, 14);
      if (c) applyCycle(this.lattice, c);
    }
    const gridW = (this.lattice.cols - 1) * this.spacing;
    const gridH = (this.lattice.rows - 1) * this.spacing;
    this.offsetX = (w - gridW) * 0.5;
    this.offsetY = (h - gridH) * 0.5;
    this.activeAnim = null;
    this.draw(performance.now());
  }

  setTheme(theme) {
    this.theme = theme === "dark" ? "dark" : "light";
    this.draw(performance.now());
  }

  pauseLoop() {
    if (this.timerId !== null) {
      clearTimeout(this.timerId);
      this.timerId = null;
    }
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  scheduleNextFlip(delayMs = 1900) {
    if (this.disposed || this.reducedMotion || !this.visible || document.hidden) return;
    if (this.timerId !== null) clearTimeout(this.timerId);
    this.timerId = setTimeout(() => {
      this.timerId = null;
      this.startFlip(null);
    }, delayMs);
  }

  handleClick(ev) {
    if (!this.lattice || this.reducedMotion) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left;
    const py = ev.clientY - rect.top;
    const gx = Math.round((px - this.offsetX) / this.spacing);
    const gy = Math.round((py - this.offsetY) / this.spacing);
    if (gx >= 0 && gx < this.lattice.cols && gy >= 0 && gy < this.lattice.rows) {
      this.startFlip(gy * this.lattice.cols + gx);
    }
  }

  startFlip(startIdx) {
    if (this.disposed || !this.lattice) return;
    if (this.activeAnim) {
      applyCycle(this.lattice, this.activeAnim);
      this.activeAnim = null;
    }
    let cycle = null;
    // Favor cycles slightly on the right / center-right on wide screens where the lattice shines.
    for (let attempt = 0; attempt < 8 && !cycle; attempt++) {
      let seed = startIdx;
      if (seed === null && this.width >= 900) {
        const minX = Math.floor(this.lattice.cols * 0.35);
        const x = minX + Math.floor(Math.random() * (this.lattice.cols - minX));
        const y = Math.floor(Math.random() * this.lattice.rows);
        seed = y * this.lattice.cols + x;
      }
      cycle = findAlternatingCycle(this.lattice, seed, 8);
    }
    if (!cycle) {
      this.scheduleNextFlip(1600);
      return;
    }
    this.activeAnim = {
      oldPairs: cycle.oldPairs,
      newPairs: cycle.newPairs,
      oldSet: new Set(cycle.oldPairs.map(([u, v]) => (u < v ? `${u}:${v}` : `${v}:${u}`))),
      startMs: performance.now(),
      durationMs: 540,
    };
    this.tick();
  }

  tick() {
    if (this.disposed) return;
    const now = performance.now();
    this.draw(now);
    if (this.activeAnim) {
      const elapsed = now - this.activeAnim.startMs;
      if (elapsed >= this.activeAnim.durationMs) {
        applyCycle(this.lattice, this.activeAnim);
        this.activeAnim = null;
        this.draw(now);
        this.rafId = null;
        this.scheduleNextFlip(1900);
        return;
      }
      this.rafId = requestAnimationFrame(() => this.tick());
    }
  }

  siteXY(idx) {
    const x = idx % this.lattice.cols;
    const y = (idx - x) / this.lattice.cols;
    return [this.offsetX + x * this.spacing, this.offsetY + y * this.spacing];
  }

  drawDimer(ctx, u, v, alpha, scale, palette) {
    if (alpha <= 0.01) return;
    const [x1, y1] = this.siteXY(u);
    const [x2, y2] = this.siteXY(v);
    const horizontal = Math.abs(y1 - y2) < 1;
    const mx = (x1 + x2) * 0.5;
    const my = (y1 + y2) * 0.5;
    const dx = (x2 - x1) * 0.5 * scale;
    const dy = (y2 - y1) * 0.5 * scale;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineCap = "round";
    const rodWidth = Math.max(5.5, this.spacing * 0.165);
    ctx.lineWidth = rodWidth;
    if (horizontal) {
      // Alternate subtly between URochester Navy and Azure based on parity row for quiet depth.
      const row = Math.floor(u / this.lattice.cols);
      ctx.strokeStyle = row % 3 === 0 ? palette.dimerHSoft : palette.dimerH;
      ctx.beginPath();
      ctx.moveTo(mx - dx, my - dy);
      ctx.lineTo(mx + dx, my + dy);
      ctx.stroke();
    } else {
      ctx.strokeStyle = palette.dimerV;
      ctx.beginPath();
      ctx.moveTo(mx - dx, my - dy);
      ctx.lineTo(mx + dx, my + dy);
      ctx.stroke();
      // Subtle luminous Dandelion inner spine in light mode so gold feels authentically yellow yet crisp.
      if (this.theme === "light") {
        ctx.lineWidth = rodWidth * 0.42;
        ctx.strokeStyle = palette.dimerVInner;
        ctx.beginPath();
        ctx.moveTo(mx - dx * 0.92, my - dy * 0.92);
        ctx.lineTo(mx + dx * 0.92, my + dy * 0.92);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  draw(now) {
    if (!this.ctx || !this.lattice) return;
    const ctx = this.ctx;
    const p = PALETTES[this.theme];
    const { cols, rows, partner } = this.lattice;

    ctx.save();
    ctx.scale(this.dpr, this.dpr);
    ctx.fillStyle = p.bg;
    ctx.fillRect(0, 0, this.width, this.height);

    // 1. Hairline lattice skeleton
    ctx.strokeStyle = p.wire;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let y = 0; y < rows; y++) {
      const py = this.offsetY + y * this.spacing;
      ctx.moveTo(this.offsetX, py);
      ctx.lineTo(this.offsetX + (cols - 1) * this.spacing, py);
    }
    for (let x = 0; x < cols; x++) {
      const px = this.offsetX + x * this.spacing;
      ctx.moveTo(px, this.offsetY);
      ctx.lineTo(px, this.offsetY + (rows - 1) * this.spacing);
    }
    ctx.stroke();

    // 2. Highlight active alternating-cycle path quietly underneath
    let progress = 0;
    if (this.activeAnim) {
      progress = Math.min(1, Math.max(0, (now - this.activeAnim.startMs) / this.activeAnim.durationMs));
      const pulse = Math.sin(progress * Math.PI);
      ctx.save();
      ctx.strokeStyle = p.cycleRing;
      ctx.lineWidth = Math.max(12, this.spacing * 0.34);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalAlpha = pulse * 0.85;
      ctx.beginPath();
      for (const [u, v] of this.activeAnim.oldPairs) {
        const [x1, y1] = this.siteXY(u);
        const [x2, y2] = this.siteXY(v);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      for (const [u, v] of this.activeAnim.newPairs) {
        const [x1, y1] = this.siteXY(u);
        const [x2, y2] = this.siteXY(v);
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
      ctx.restore();
    }

    // 3. Static dimers (all pairs not currently flipping)
    const activeOld = this.activeAnim ? this.activeAnim.oldSet : null;
    for (let idx = 0; idx < this.lattice.total; idx++) {
      const q = partner[idx];
      if (idx >= q) continue;
      if (activeOld && activeOld.has(`${idx}:${q}`)) continue;
      this.drawDimer(ctx, idx, q, 0.92, 0.92, p);
    }

    // 4. Flipping alternating-cycle dimers (smooth cross-fade & slight settle)
    if (this.activeAnim) {
      const eased = easeOutCubic(progress);
      for (const [u, v] of this.activeAnim.oldPairs) {
        this.drawDimer(ctx, u, v, (1 - eased) * 0.92, 0.92 - eased * 0.16, p);
      }
      for (const [u, v] of this.activeAnim.newPairs) {
        this.drawDimer(ctx, u, v, eased * 0.96, 0.76 + eased * 0.16, p);
      }
    }

    // 5. Crisp bipartite parity nodes (● even / ○ odd)
    const nodeR = Math.max(2.6, this.spacing * 0.068);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const px = this.offsetX + x * this.spacing;
        const py = this.offsetY + y * this.spacing;
        ctx.beginPath();
        ctx.arc(px, py, nodeR, 0, Math.PI * 2);
        if ((x + y) % 2 === 0) {
          ctx.fillStyle = p.rowNode;
          ctx.fill();
        } else {
          ctx.fillStyle = p.colFill;
          ctx.fill();
          ctx.lineWidth = 1.3;
          ctx.strokeStyle = p.colStroke;
          ctx.stroke();
        }
      }
    }

    ctx.restore();
  }

  dispose() {
    this.disposed = true;
    this.pauseLoop();
    this.resizeObserver.disconnect();
    this.intersectionObserver.disconnect();
    this.canvas.removeEventListener("click", this.onClick);
    document.removeEventListener("visibilitychange", this.onVisibility);
  }
}

let activeHero = null;

window.PermanentHero = {
  mount(canvas, theme) {
    if (!canvas) return;
    if (activeHero && activeHero.canvas === canvas) {
      activeHero.setTheme(theme);
      return;
    }
    if (activeHero) {
      activeHero.dispose();
      activeHero = null;
    }
    activeHero = new HeroScene(canvas, theme);
  },
  setTheme(theme) {
    if (activeHero) activeHero.setTheme(theme);
  },
  triggerFlip() {
    if (activeHero) activeHero.startFlip(null);
  },
  dispose() {
    if (activeHero) {
      activeHero.dispose();
      activeHero = null;
    }
  },
};
