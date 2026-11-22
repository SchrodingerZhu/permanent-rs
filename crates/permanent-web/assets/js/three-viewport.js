// Restrained, architectural Three.js 3D cubic dimer viewport.
// Uses self-vendored Three.js 0.187.0 + OrbitControls, InstancedMesh for all
// sites and dimers, Z-layer slicing, Ortho/Persp projection toggle, and
// strictly on-demand rendering (zero idle GPU work).

import {
  AmbientLight,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DirectionalLight,
  Float32BufferAttribute,
  Group,
  HemisphereLight,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  MeshBasicMaterial,
  MeshStandardMaterial,
  OrbitControls,
  OrthographicCamera,
  PerspectiveCamera,
  Quaternion,
  Scene,
  SphereGeometry,
  Vector3,
  WebGLRenderer,
} from "./vendor/three-bundle.min.js";

const THEMES = {
  light: {
    bg: 0xffffff,
    wire: 0xcbd5e1,
    wireOpacity: 0.72,
    wrapWire: 0x94a3b8,
    rowNode: 0x001e5f, // URochester Navy (●)
    colCore: 0xffffff, // hollow white core (○)
    colRim: 0x001e5f,
    axisColors: [
      new Color(0x001e5f), // X: URochester Navy
      new Color(0x0066fd), // Y: Arpeggio Azure
      new Color(0xe09b00), // Z: Luminous Rochester Gold
    ],
    invalidColor: new Color(0xd9483b),
    fadedColor: new Color(0xdbe3ef),
  },
  dark: {
    bg: 0x0c1736,
    wire: 0x2a427e,
    wireOpacity: 0.78,
    wrapWire: 0x3b589e,
    rowNode: 0xe6eeff, // bright filled (●)
    colCore: 0x0c1736, // dark hollow core (○)
    colRim: 0x8db8ff,
    axisColors: [
      new Color(0x66a2ff), // X: Connective Cornflower
      new Color(0xb7d3ff), // Y: Wondrous Sky
      new Color(0xffd82b), // Z: Dandelion Yellow
    ],
    invalidColor: new Color(0xff6b6b),
    fadedColor: new Color(0x1b2d5b),
  },
};

const UP = new Vector3(0, 1, 0);

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

class ThreeLatticeViewport {
  constructor(canvas, theme) {
    this.canvas = canvas;
    this.themeName = theme === "dark" ? "dark" : "light";
    this.disposed = false;

    this.scene = new Scene();
    this.scene.background = new Color(THEMES[this.themeName].bg);

    this.renderer = new WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // Cameras
    this.perspCamera = new PerspectiveCamera(34, 1, 0.1, 200);
    this.orthoCamera = new OrthographicCamera(-5, 5, 5, -5, 0.1, 200);
    this.projection = "ortho";
    this.activeCamera = this.orthoCamera;

    // Clean balanced studio lighting
    this.hemiLight = new HemisphereLight(0xffffff, 0xd4def2, 1.05);
    this.scene.add(this.hemiLight);
    this.ambientLight = new AmbientLight(0xffffff, 0.45);
    this.scene.add(this.ambientLight);
    this.keyLight = new DirectionalLight(0xffffff, 0.9);
    this.keyLight.position.set(10, 16, 12);
    this.scene.add(this.keyLight);
    this.fillLight = new DirectionalLight(0xdce8ff, 0.48);
    this.fillLight.position.set(-10, -6, -8);
    this.scene.add(this.fillLight);

    // Controls
    this.controls = new OrbitControls(this.activeCamera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.085;
    this.controls.rotateSpeed = 0.72;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.7;

    // Scene data
    this.rootGroup = new Group();
    this.scene.add(this.rootGroup);
    this.geometry = null; // Lattice3d JSON
    this.bondLookup = new Map(); // "u:v" -> bond info
    this.matching = null;
    this.matchingValid = true;
    this.sliceZ = null;
    this.showPeriodic = true;
    this.radiusExtent = 4;

    // Meshes
    this.wireMesh = null;
    this.wrapWireMesh = null;
    this.rowMesh = null;
    this.colCoreMesh = null;
    this.colRimMesh = null;
    this.dimerMesh = null;

    // Shared unit geometries
    this.unitSphere = new SphereGeometry(1, 18, 14);
    this.unitCylinder = new CylinderGeometry(1, 1, 1, 18, 1, false);

    // On-demand render & short animation tracking
    this.rafId = null;
    this.dampingFrames = 0;
    this.cameraTween = null;

    this.controls.addEventListener("change", () => this.requestRender());
    this.controls.addEventListener("start", () => {
      this.cameraTween = null;
      this.dampingFrames = 18;
      this.requestRender();
    });
    this.controls.addEventListener("end", () => {
      this.dampingFrames = 24;
      this.requestRender();
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);

    this.resize();
    this.applyCameraPreset("iso", false);
  }

  centerOf(dims) {
    return new Vector3((dims[0] - 1) * 0.5, (dims[1] - 1) * 0.5, (dims[2] - 1) * 0.5);
  }

  siteToVec(coord, dims) {
    // Map lattice (x, y, z) cleanly so lattice z points upward visually for intuitive Z-slicing.
    const c = this.centerOf(dims);
    return new Vector3(coord[0] - c.x, coord[2] - c.z, coord[1] - c.y);
  }

  resize() {
    if (this.disposed) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(240, Math.floor(rect.width));
    const h = Math.max(240, Math.floor(rect.height));
    this.renderer.setSize(w, h, false);
    this.aspect = w / h;
    this.updateCameraFrustum();
    this.requestRender();
  }

  updateCameraFrustum() {
    const aspect = this.aspect || 1;
    this.perspCamera.aspect = aspect;
    this.perspCamera.updateProjectionMatrix();

    const halfH = this.radiusExtent * 0.94;
    const halfW = halfH * aspect;
    this.orthoCamera.left = -halfW;
    this.orthoCamera.right = halfW;
    this.orthoCamera.top = halfH;
    this.orthoCamera.bottom = -halfH;
    this.orthoCamera.updateProjectionMatrix();
  }

  clearSceneObjects() {
    while (this.rootGroup.children.length > 0) {
      const child = this.rootGroup.children[0];
      this.rootGroup.remove(child);
      if (child.geometry && child.geometry !== this.unitSphere && child.geometry !== this.unitCylinder) {
        child.geometry.dispose();
      }
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
        else child.material.dispose();
      }
    }
    this.wireMesh = null;
    this.wrapWireMesh = null;
    this.rowMesh = null;
    this.colCoreMesh = null;
    this.colRimMesh = null;
    this.dimerMesh = null;
  }

  setProblem(geometry, matching, valid = true) {
    if (this.disposed || !geometry || geometry.kind !== "lattice3d") return;
    const prevDims = this.geometry ? this.geometry.dims.join("x") : null;
    this.geometry = geometry;
    this.matching = matching || null;
    this.matchingValid = valid !== false;

    const dims = geometry.dims;
    const maxDim = Math.max(dims[0], dims[1], dims[2]);
    this.radiusExtent = Math.max(2.0, Math.sqrt(dims[0] * dims[0] + dims[1] * dims[1] + dims[2] * dims[2]) * 0.56);

    // Index bonds for O(1) lookup when rendering matchings
    this.bondLookup.clear();
    let minW = Infinity;
    let maxW = 0;
    for (const b of geometry.bonds) {
      if (b.weight > 0) {
        minW = Math.min(minW, b.weight);
        maxW = Math.max(maxW, b.weight);
      }
      const key = `${b.row}:${b.col}`;
      // Prefer non-wrapping bond if multiple exist between same pair (e.g. L=2)
      if (!this.bondLookup.has(key) || !b.periodic_wrap) {
        this.bondLookup.set(key, b);
      }
    }
    this.minWeight = Number.isFinite(minW) ? minW : 1;
    this.maxWeight = maxW > 0 ? maxW : 1;

    if (this.sliceZ !== null && this.sliceZ >= dims[2]) {
      this.sliceZ = null;
    }

    this.rebuildMeshes();
    this.updateCameraFrustum();

    if (prevDims !== dims.join("x")) {
      this.applyCameraPreset("iso", false);
    }
    this.requestRender();
  }

  rebuildMeshes() {
    this.clearSceneObjects();
    if (!this.geometry) return;
    const p = THEMES[this.themeName];
    const { dims, row_sites, col_sites, bonds } = this.geometry;
    const maxDim = Math.max(dims[0], dims[1], dims[2]);
    const densityScale = Math.max(0.55, Math.min(1.0, 4.6 / Math.max(3, maxDim)));

    // 1. Hairline lattice skeleton (interior + periodic stubs)
    const interiorVerts = [];
    const wrapVerts = [];
    for (const b of bonds) {
      const aCoord = row_sites[b.row];
      const bCoord = col_sites[b.col];
      if (!aCoord || !bCoord) continue;
      if (this.sliceZ !== null && aCoord[2] !== this.sliceZ && bCoord[2] !== this.sliceZ) {
        continue;
      }
      const va = this.siteToVec(aCoord, dims);
      const vb = this.siteToVec(bCoord, dims);
      if (!b.periodic_wrap) {
        interiorVerts.push(va.x, va.y, va.z, vb.x, vb.y, vb.z);
      } else {
        // Draw short crisp boundary stubs pointing outward along the wrap axis
        const dir = new Vector3();
        if (b.axis === 0) dir.set(aCoord[0] === 0 ? -1 : 1, 0, 0);
        else if (b.axis === 1) dir.set(0, 0, aCoord[1] === 0 ? -1 : 1);
        else dir.set(0, aCoord[2] === 0 ? -1 : 1, 0);
        const stub = 0.36;
        wrapVerts.push(va.x, va.y, va.z, va.x + dir.x * stub, va.y + dir.y * stub, va.z + dir.z * stub);
        wrapVerts.push(vb.x, vb.y, vb.z, vb.x - dir.x * stub, vb.y - dir.y * stub, vb.z - dir.z * stub);
      }
    }

    if (interiorVerts.length > 0) {
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(interiorVerts, 3));
      const m = new LineBasicMaterial({
        color: p.wire,
        transparent: true,
        opacity: p.wireOpacity,
      });
      this.wireMesh = new LineSegments(g, m);
      this.rootGroup.add(this.wireMesh);
    }

    if (wrapVerts.length > 0) {
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(wrapVerts, 3));
      const m = new LineBasicMaterial({
        color: p.wrapWire,
        transparent: true,
        opacity: 0.55,
      });
      this.wrapWireMesh = new LineSegments(g, m);
      this.wrapWireMesh.visible = this.showPeriodic;
      this.rootGroup.add(this.wrapWireMesh);
    }

    // 2. Bipartite site spheres (● filled row sites vs ○ hollow ringed col sites)
    const nodeRadius = 0.118 * densityScale;
    const rowMat = new MeshStandardMaterial({ color: p.rowNode, roughness: 0.42, metalness: 0.04 });
    this.rowMesh = new InstancedMesh(this.unitSphere, rowMat, row_sites.length);

    const colCoreMat = new MeshBasicMaterial({ color: p.colCore });
    this.colCoreMesh = new InstancedMesh(this.unitSphere, colCoreMat, col_sites.length);

    // BackSide (side = 1) outer shell creates a crisp 2D-style hollow ring (○) from every angle.
    const colRimMat = new MeshBasicMaterial({ color: p.colRim, side: 1 });
    this.colRimMesh = new InstancedMesh(this.unitSphere, colRimMat, col_sites.length);

    const mat = new Matrix4();
    const scaleVec = new Vector3();
    const quat = new Quaternion();

    for (let i = 0; i < row_sites.length; i++) {
      const c = row_sites[i];
      const pos = this.siteToVec(c, dims);
      const r = this.sliceZ !== null && c[2] !== this.sliceZ ? nodeRadius * 0.28 : nodeRadius;
      scaleVec.set(r, r, r);
      mat.compose(pos, quat, scaleVec);
      this.rowMesh.setMatrixAt(i, mat);
    }
    this.rowMesh.instanceMatrix.needsUpdate = true;
    this.rootGroup.add(this.rowMesh);

    for (let i = 0; i < col_sites.length; i++) {
      const c = col_sites[i];
      const pos = this.siteToVec(c, dims);
      const inSlice = this.sliceZ === null || c[2] === this.sliceZ;
      const r = inSlice ? nodeRadius : nodeRadius * 0.28;
      scaleVec.set(r * 1.24, r * 1.24, r * 1.24);
      mat.compose(pos, quat, scaleVec);
      this.colRimMesh.setMatrixAt(i, mat);

      scaleVec.set(r * 0.92, r * 0.92, r * 0.92);
      mat.compose(pos, quat, scaleVec);
      this.colCoreMesh.setMatrixAt(i, mat);
    }
    this.colRimMesh.instanceMatrix.needsUpdate = true;
    this.colCoreMesh.instanceMatrix.needsUpdate = true;
    this.rootGroup.add(this.colRimMesh);
    this.rootGroup.add(this.colCoreMesh);

    // 3. Active dimer rods (InstancedMesh, up to 2 segments per periodic wrap dimer)
    const maxInstances = Math.max(1, row_sites.length * 2);
    const dimerMat = new MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.34,
      metalness: 0.04,
    });
    this.dimerMesh = new InstancedMesh(this.unitCylinder, dimerMat, maxInstances);
    this.rootGroup.add(this.dimerMesh);
    this.updateDimerInstances(densityScale);
  }

  updateDimerInstances(densityScale = null) {
    if (!this.dimerMesh || !this.geometry) return;
    const p = THEMES[this.themeName];
    const { dims, row_sites, col_sites } = this.geometry;
    if (densityScale === null) {
      const maxDim = Math.max(dims[0], dims[1], dims[2]);
      densityScale = Math.max(0.55, Math.min(1.0, 4.6 / Math.max(3, maxDim)));
    }
    const baseRadius = 0.095 * densityScale;
    const matching = this.matching;
    if (!matching || matching.length === 0) {
      this.dimerMesh.count = 0;
      return;
    }

    const mat = new Matrix4();
    const quat = new Quaternion();
    const mid = new Vector3();
    const delta = new Vector3();
    const dir = new Vector3();
    const scaleVec = new Vector3();

    const hasWeightSpread = this.maxWeight / Math.max(this.minWeight, 1e-12) > 1.05;
    const logSpan = hasWeightSpread ? Math.log(this.maxWeight / this.minWeight) : 1;

    let count = 0;
    for (let u = 0; u < matching.length && u < row_sites.length; u++) {
      const v = matching[u];
      const aCoord = row_sites[u];
      const bCoord = col_sites[v];
      if (!aCoord || !bCoord) continue;

      const bond = this.bondLookup.get(`${u}:${v}`);
      const inSlice = this.sliceZ === null || aCoord[2] === this.sliceZ || bCoord[2] === this.sliceZ;
      if (this.sliceZ !== null && !inSlice) continue;

      let color = p.invalidColor;
      let radius = baseRadius * 0.72;
      if (bond) {
        color = p.axisColors[bond.axis % 3];
        if (hasWeightSpread && bond.weight > 0) {
          const norm = Math.log(bond.weight / this.minWeight) / logSpan;
          radius = baseRadius * (0.76 + 0.48 * Math.min(1, Math.max(0, norm)));
        } else {
          radius = baseRadius;
        }
      }

      const va = this.siteToVec(aCoord, dims);
      const vb = this.siteToVec(bCoord, dims);

      if (bond && bond.periodic_wrap) {
        if (!this.showPeriodic) continue;
        // Two crisp half-rods sticking outward from the wrapped boundary sites
        const axisDir = new Vector3();
        if (bond.axis === 0) axisDir.set(aCoord[0] === 0 ? -1 : 1, 0, 0);
        else if (bond.axis === 1) axisDir.set(0, 0, aCoord[1] === 0 ? -1 : 1);
        else axisDir.set(0, aCoord[2] === 0 ? -1 : 1, 0);

        const segLen = 0.44;
        quat.setFromUnitVectors(UP, axisDir);
        scaleVec.set(radius, segLen, radius);

        mid.copy(va).addScaledVector(axisDir, segLen * 0.5);
        mat.compose(mid, quat, scaleVec);
        this.dimerMesh.setMatrixAt(count, mat);
        this.dimerMesh.setColorAt(count, color);
        count++;

        mid.copy(vb).addScaledVector(axisDir, -segLen * 0.5);
        mat.compose(mid, quat, scaleVec);
        this.dimerMesh.setMatrixAt(count, mat);
        this.dimerMesh.setColorAt(count, color);
        count++;
      } else {
        delta.subVectors(vb, va);
        const len = delta.length();
        if (len < 1e-4) continue;
        dir.copy(delta).divideScalar(len);
        quat.setFromUnitVectors(UP, dir);
        mid.addVectors(va, vb).multiplyScalar(0.5);
        scaleVec.set(radius, len, radius);
        mat.compose(mid, quat, scaleVec);
        this.dimerMesh.setMatrixAt(count, mat);
        this.dimerMesh.setColorAt(count, color);
        count++;
      }
    }

    this.dimerMesh.count = count;
    this.dimerMesh.instanceMatrix.needsUpdate = true;
    if (this.dimerMesh.instanceColor) {
      this.dimerMesh.instanceColor.needsUpdate = true;
    }
  }

  setMatching(matching, valid = true) {
    if (this.disposed) return;
    this.matching = matching;
    this.matchingValid = valid !== false;
    this.updateDimerInstances();
    this.requestRender();
  }

  setTheme(theme) {
    const next = theme === "dark" ? "dark" : "light";
    if (next === this.themeName) return;
    this.themeName = next;
    this.scene.background = new Color(THEMES[this.themeName].bg);
    this.rebuildMeshes();
    this.requestRender();
  }

  setProjection(mode) {
    const next = mode === "persp" ? "persp" : "ortho";
    if (next === this.projection) return;
    const prevPos = this.activeCamera.position.clone();
    const target = this.controls.target.clone();
    this.projection = next;
    this.activeCamera = next === "persp" ? this.perspCamera : this.orthoCamera;
    this.activeCamera.position.copy(prevPos);
    this.activeCamera.lookAt(target);
    this.controls.object = this.activeCamera;
    this.controls.target.copy(target);
    this.orthoCamera.zoom = 1;
    this.updateCameraFrustum();
    this.controls.update();
    this.requestRender();
  }

  setSliceZ(zOrNull) {
    const next = zOrNull === null || zOrNull === undefined || zOrNull < 0 ? null : Number(zOrNull);
    if (next === this.sliceZ) return;
    this.sliceZ = next;
    this.rebuildMeshes();
    this.requestRender();
  }

  setShowPeriodic(show) {
    this.showPeriodic = Boolean(show);
    if (this.wrapWireMesh) {
      this.wrapWireMesh.visible = this.showPeriodic;
    }
    this.updateDimerInstances();
    this.requestRender();
  }

  applyCameraPreset(preset, animate = true) {
    const dist = this.radiusExtent * 3.1;
    let targetPos;
    if (preset === "front") {
      targetPos = new Vector3(0, 0, dist);
    } else if (preset === "top") {
      targetPos = new Vector3(0.01, dist, 0.01);
    } else {
      // Classic clean architectural isometric angle
      targetPos = new Vector3(1.25, 0.92, 1.38).normalize().multiplyScalar(dist);
    }
    const targetLook = new Vector3(0, 0, 0);

    if (!animate) {
      this.cameraTween = null;
      this.perspCamera.position.copy(targetPos);
      this.orthoCamera.position.copy(targetPos);
      this.orthoCamera.zoom = 1;
      this.orthoCamera.updateProjectionMatrix();
      this.controls.target.copy(targetLook);
      this.controls.update();
      this.requestRender();
      return;
    }

    this.cameraTween = {
      fromPos: this.activeCamera.position.clone(),
      toPos: targetPos,
      fromTarget: this.controls.target.clone(),
      toTarget: targetLook,
      fromZoom: this.orthoCamera.zoom,
      toZoom: 1,
      startMs: performance.now(),
      durationMs: 240,
    };
    this.requestRender();
  }

  requestRender() {
    if (this.disposed || this.rafId !== null) return;
    this.rafId = requestAnimationFrame(() => this.renderFrame());
  }

  renderFrame() {
    this.rafId = null;
    if (this.disposed) return;

    let keepGoing = false;
    if (this.cameraTween) {
      const t = Math.min(1, (performance.now() - this.cameraTween.startMs) / this.cameraTween.durationMs);
      const e = easeOutCubic(t);
      this.activeCamera.position.lerpVectors(this.cameraTween.fromPos, this.cameraTween.toPos, e);
      this.controls.target.lerpVectors(this.cameraTween.fromTarget, this.cameraTween.toTarget, e);
      if (this.projection === "ortho") {
        this.orthoCamera.zoom = this.cameraTween.fromZoom + (this.cameraTween.toZoom - this.cameraTween.fromZoom) * e;
        this.orthoCamera.updateProjectionMatrix();
      }
      if (t >= 1) {
        this.cameraTween = null;
      } else {
        keepGoing = true;
      }
    }

    this.controls.update();
    if (this.dampingFrames > 0) {
      this.dampingFrames--;
      keepGoing = true;
    }

    this.renderer.render(this.scene, this.activeCamera);
    if (keepGoing) {
      this.requestRender();
    }
  }

  dispose() {
    this.disposed = true;
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.resizeObserver.disconnect();
    this.clearSceneObjects();
    this.unitSphere.dispose();
    this.unitCylinder.dispose();
    this.controls.dispose();
    this.renderer.dispose();
  }
}

let activeViewport = null;

window.PermanentThree = {
  mount(canvas, theme) {
    if (!canvas) return;
    if (activeViewport && activeViewport.canvas === canvas) {
      activeViewport.setTheme(theme);
      return;
    }
    if (activeViewport) {
      activeViewport.dispose();
      activeViewport = null;
    }
    activeViewport = new ThreeLatticeViewport(canvas, theme);
  },
  setProblem(geometry, matching, valid) {
    if (activeViewport) activeViewport.setProblem(geometry, matching, valid);
  },
  setMatching(matching, valid) {
    if (activeViewport) activeViewport.setMatching(matching, valid);
  },
  setTheme(theme) {
    if (activeViewport) activeViewport.setTheme(theme);
  },
  setViewPreset(name) {
    if (activeViewport) activeViewport.applyCameraPreset(name, true);
  },
  setProjection(mode) {
    if (activeViewport) activeViewport.setProjection(mode);
  },
  setSliceZ(zOrNull) {
    if (activeViewport) activeViewport.setSliceZ(zOrNull);
  },
  setShowPeriodic(show) {
    if (activeViewport) activeViewport.setShowPeriodic(show);
  },
  dispose() {
    if (activeViewport) {
      activeViewport.dispose();
      activeViewport = null;
    }
  },
};
