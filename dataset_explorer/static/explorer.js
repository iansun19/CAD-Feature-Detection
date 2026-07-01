import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const GRAY_DIM = 0xbbbbbb;
const DIM_OPACITY_ISOLATE = 0.5;
const DIM_OPACITY_FOCUS = 0.1;

const state = {
  classes: [],
  selectedClassId: null,
  partsOffset: 0,
  partsTotal: 0,
  currentPartId: null,
  viewMode: "full", // full | isolate | focus
  isolatedClassId: null,
  focusClassId: null,
};

const els = {
  classList: document.getElementById("class-list"),
  galleryTitle: document.getElementById("gallery-title"),
  gallerySubtitle: document.getElementById("gallery-subtitle"),
  partGrid: document.getElementById("part-grid"),
  loadMoreBtn: document.getElementById("load-more-btn"),
  viewerWrap: document.querySelector(".viewer-wrap"),
  viewerCanvas: document.getElementById("viewer-canvas"),
  viewerLoading: document.getElementById("viewer-loading"),
  faceTooltip: document.getElementById("face-tooltip"),
  partInfo: document.getElementById("part-info"),
  viewerTitle: document.getElementById("viewer-title"),
  focusClassBtn: document.getElementById("focus-class-btn"),
  resetViewBtn: document.getElementById("reset-view-btn"),
};

const viewer = {
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  root: null,
  faceMeshes: [],
  raycaster: new THREE.Raycaster(),
  pointer: new THREE.Vector2(),
  animId: null,
};

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function classById(id) {
  return state.classes.find((c) => c.id === id);
}

function hexToThree(hex) {
  return new THREE.Color(hex);
}

function renderClassList() {
  els.classList.innerHTML = "";
  for (const cls of state.classes) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "class-item" + (state.selectedClassId === cls.id ? " selected" : "");
    item.dataset.classId = cls.id;

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = cls.color;

    const meta = document.createElement("div");
    meta.className = "class-meta";
    meta.innerHTML = `
      <div class="name">${cls.id} · ${cls.name}</div>
      <div class="stats">${cls.part_count.toLocaleString()} parts · ${cls.face_count_total.toLocaleString()} faces</div>
      <div class="desc">${cls.description}</div>
    `;

    item.appendChild(swatch);
    item.appendChild(meta);
    item.addEventListener("click", () => selectClass(cls.id));
    els.classList.appendChild(item);
  }
}

async function selectClass(classId) {
  state.selectedClassId = classId;
  state.partsOffset = 0;
  state.focusClassId = null;
  state.viewMode = "full";
  state.isolatedClassId = null;
  renderClassList();

  const cls = classById(classId);
  els.galleryTitle.textContent = `${cls.id} · ${cls.name}`;
  els.gallerySubtitle.textContent = cls.description;
  els.focusClassBtn.classList.remove("hidden");
  els.focusClassBtn.textContent = `Focus on class ${classId}`;

  await loadPartsPage(true);
}

async function loadPartsPage(reset = false) {
  if (state.selectedClassId == null) return;
  if (reset) {
    state.partsOffset = 0;
    els.partGrid.innerHTML = '<p class="loading">Loading parts…</p>';
  }

  const data = await fetchJson(
    `/api/classes/${state.selectedClassId}/parts?limit=10&offset=${state.partsOffset}`
  );
  state.partsTotal = data.total;

  if (reset) els.partGrid.innerHTML = "";

  if (data.parts.length === 0 && state.partsOffset === 0) {
    els.partGrid.innerHTML = '<p class="muted">No parts found for this class.</p>';
  } else {
    for (const part of data.parts) {
      els.partGrid.appendChild(makePartCard(part));
    }
  }

  state.partsOffset += data.parts.length;
  const hasMore = state.partsOffset < state.partsTotal;
  els.loadMoreBtn.classList.toggle("hidden", !hasMore);
}

function makePartCard(part) {
  const card = document.createElement("div");
  card.className = "part-card";
  card.innerHTML = `
    <div class="part-id">${part.part_id}</div>
    <div class="face-count">${part.face_count} face${part.face_count === 1 ? "" : "s"} of this class</div>
  `;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn";
  btn.textContent = "View 3D";
  btn.addEventListener("click", () => loadPart(part.part_id));
  card.appendChild(btn);
  return card;
}

function viewerSize() {
  const wrap = els.viewerWrap;
  return {
    w: Math.max(wrap?.clientWidth || 640, 1),
    h: Math.max(wrap?.clientHeight || 480, 1),
  };
}

function initViewer() {
  const { w, h } = viewerSize();

  viewer.scene = new THREE.Scene();
  viewer.scene.background = new THREE.Color(0xffffff);

  viewer.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100000);
  viewer.camera.position.set(120, 90, 120);

  viewer.renderer = new THREE.WebGLRenderer({ antialias: true });
  viewer.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewer.renderer.setSize(w, h);
  els.viewerCanvas.innerHTML = "";
  els.viewerCanvas.appendChild(viewer.renderer.domElement);

  viewer.controls = new OrbitControls(viewer.camera, viewer.renderer.domElement);
  viewer.controls.enableDamping = true;
  viewer.controls.dampingFactor = 0.08;
  viewer.controls.enableZoom = true;
  viewer.controls.zoomSpeed = 1.4;
  viewer.controls.enablePan = true;
  viewer.controls.screenSpacePanning = true;
  viewer.controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };

  const canvas = viewer.renderer.domElement;
  canvas.addEventListener("wheel", (e) => e.preventDefault(), { passive: false });
  canvas.addEventListener("pointerdown", onPointerDown);

  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  const hemi = new THREE.HemisphereLight(0xffffff, 0xb0b0b0, 0.45);
  const dir1 = new THREE.DirectionalLight(0xffffff, 0.85);
  dir1.position.set(1, 1.2, 0.8);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dir2.position.set(-0.8, -0.4, -1);
  viewer.scene.add(ambient, hemi, dir1, dir2);

  viewer.root = new THREE.Group();
  viewer.scene.add(viewer.root);

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => onResize()).observe(els.viewerWrap);
  } else {
    window.addEventListener("resize", onResize);
  }

  animate();
}

function onResize() {
  if (!viewer.renderer) return;
  const { w, h } = viewerSize();
  viewer.camera.aspect = w / h;
  viewer.camera.updateProjectionMatrix();
  viewer.renderer.setSize(w, h);
}

function animate() {
  viewer.animId = requestAnimationFrame(animate);
  viewer.controls?.update();
  viewer.renderer?.render(viewer.scene, viewer.camera);
}

function clearViewerMeshes() {
  for (const mesh of viewer.faceMeshes) {
    mesh.geometry.dispose();
    mesh.material.dispose();
    viewer.root.remove(mesh);
  }
  viewer.faceMeshes = [];
}

function applyViewMode() {
  for (const mesh of viewer.faceMeshes) {
    const cls = mesh.userData.classId;
    const baseColor = hexToThree(mesh.userData.baseColor);
    const mat = mesh.material;

    if (state.viewMode === "full") {
      mat.color.copy(baseColor);
      mat.opacity = 1;
      mat.transparent = false;
    } else if (state.viewMode === "isolate") {
      if (cls === state.isolatedClassId) {
        mat.color.copy(baseColor);
        mat.opacity = 1;
        mat.transparent = false;
      } else {
        mat.color.setHex(GRAY_DIM);
        mat.opacity = DIM_OPACITY_ISOLATE;
        mat.transparent = true;
      }
    } else if (state.viewMode === "focus") {
      if (cls === state.focusClassId) {
        mat.color.copy(baseColor);
        mat.opacity = 1;
        mat.transparent = false;
      } else {
        mat.color.copy(baseColor);
        mat.opacity = DIM_OPACITY_FOCUS;
        mat.transparent = true;
      }
    }
    mat.needsUpdate = true;
  }
}

function buildFaceMesh(face) {
  const positions = [];
  for (const v of face.triangles) {
    positions.push(v[0], v[1], v[2]);
  }
  if (positions.length === 0) return null;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.computeVertexNormals();

  const cls = classById(face.class_id);
  const colorHex = cls?.color || "#888888";
  const mat = new THREE.MeshStandardMaterial({
    color: hexToThree(colorHex),
    metalness: 0.05,
    roughness: 0.72,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geom, mat);
  mesh.userData = {
    faceIndex: face.face_index,
    classId: face.class_id,
    className: face.class_name,
    baseColor: colorHex,
  };

  return mesh;
}

function centerAndScale(root) {
  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const scale = 100 / maxDim;

  root.position.sub(center);
  root.scale.setScalar(scale);
  return maxDim * scale;
}

function fitCameraToModel() {
  const box = new THREE.Box3().setFromObject(viewer.root);
  if (box.isEmpty()) return;

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const fitHeight = maxDim / (2 * Math.tan((viewer.camera.fov * Math.PI) / 360));
  const fitWidth = fitHeight * viewer.camera.aspect;
  const distance = Math.max(fitHeight, fitWidth) * 1.35;

  viewer.controls.target.copy(center);
  viewer.camera.position.set(
    center.x + distance * 0.75,
    center.y + distance * 0.55,
    center.z + distance * 0.75
  );

  viewer.controls.minDistance = maxDim * 0.02;
  viewer.controls.maxDistance = maxDim * 80;
  viewer.camera.near = Math.max(maxDim / 500, 0.001);
  viewer.camera.far = maxDim * 200;
  viewer.camera.updateProjectionMatrix();
  viewer.controls.update();
}

async function loadPart(partId) {
  state.currentPartId = partId;
  state.viewMode = "full";
  state.isolatedClassId = null;
  state.focusClassId = null;
  els.faceTooltip.classList.add("hidden");
  els.viewerTitle.textContent = `3D Viewer · ${partId}`;
  els.viewerLoading.classList.remove("hidden");
  els.resetViewBtn.classList.remove("hidden");

  try {
    const [geom, info] = await Promise.all([
      fetchJson(`/api/parts/${partId}/geometry`),
      fetchJson(`/api/parts/${partId}/info`),
    ]);

    clearViewerMeshes();

    for (const face of geom.faces) {
      const mesh = buildFaceMesh(face);
      if (mesh) {
        viewer.root.add(mesh);
        viewer.faceMeshes.push(mesh);
      }
    }

    centerAndScale(viewer.root);
    fitCameraToModel();

    applyViewMode();
    renderPartInfo(info);
  } catch (err) {
    els.partInfo.classList.remove("hidden");
    els.partInfo.textContent = `Error loading ${partId}: ${err.message}`;
  } finally {
    els.viewerLoading.classList.add("hidden");
  }
}

function renderPartInfo(info) {
  const entries = Object.entries(info.class_distribution)
    .map(([id, count]) => {
      const cls = classById(Number(id));
      return { id: Number(id), name: cls?.name || id, count, color: cls?.color || "#888" };
    })
    .sort((a, b) => b.count - a.count);

  const lines = entries
    .slice(0, 8)
    .map((e) => `<span style="color:${e.color}">■</span> ${e.id} ${e.name}: ${e.count}`)
    .join(" · ");

  els.partInfo.innerHTML = `
    <strong>${info.part_id}</strong> · ${info.n_faces} faces<br>
    ${lines}${entries.length > 8 ? " · …" : ""}
  `;
  els.partInfo.classList.remove("hidden");
}

function onPointerDown(event) {
  if (!viewer.renderer || viewer.faceMeshes.length === 0) return;

  const rect = viewer.renderer.domElement.getBoundingClientRect();
  viewer.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  viewer.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  viewer.raycaster.setFromCamera(viewer.pointer, viewer.camera);
  const hits = viewer.raycaster.intersectObjects(viewer.faceMeshes, false);

  if (hits.length === 0) {
    state.viewMode = "full";
    state.isolatedClassId = null;
    els.faceTooltip.classList.add("hidden");
    applyViewMode();
    return;
  }

  const hit = hits[0].object;
  const clsId = hit.userData.classId;

  if (state.viewMode === "isolate" && state.isolatedClassId === clsId) {
    state.viewMode = "full";
    state.isolatedClassId = null;
    els.faceTooltip.classList.add("hidden");
  } else {
    state.viewMode = "isolate";
    state.isolatedClassId = clsId;
    state.focusClassId = null;
    els.faceTooltip.textContent = `Class ${clsId} · ${hit.userData.className}`;
    els.faceTooltip.classList.remove("hidden");
  }
  applyViewMode();
}

function resetView() {
  state.viewMode = "full";
  state.isolatedClassId = null;
  state.focusClassId = null;
  els.faceTooltip.classList.add("hidden");
  fitCameraToModel();
  applyViewMode();
}

function focusSelectedClass() {
  if (state.selectedClassId == null) return;
  state.viewMode = "focus";
  state.focusClassId = state.selectedClassId;
  state.isolatedClassId = null;
  els.faceTooltip.classList.add("hidden");
  applyViewMode();
}

els.loadMoreBtn.addEventListener("click", () => loadPartsPage(false));
els.focusClassBtn.addEventListener("click", focusSelectedClass);
els.resetViewBtn.addEventListener("click", resetView);

async function boot() {
  initViewer();
  state.classes = await fetchJson("/api/classes");
  renderClassList();
  if (state.classes.length > 0) {
    await selectClass(0);
  }
}

boot().catch((err) => {
  els.classList.innerHTML = `<p class="placeholder" style="color:var(--danger)">${err.message}</p>`;
});
