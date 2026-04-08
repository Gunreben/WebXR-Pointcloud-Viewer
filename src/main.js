import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { PCDLoader } from 'three/addons/loaders/PCDLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';

// ── Configuration ──────────────────────────────────────────

const POINTCLOUD_PATH = 'Pointclouds/people_neon_transformed.ply';
const AUTO_CENTER = false;  // set true to re-center clouds on load
const XR_FOVEATION = 0;
const DEFAULT_SOFT_POINTS = true;
const BG_COLOR = 0x111111;
const MOVE_SPEED = 2.5;        // meters per second (fly locomotion)
const STICK_DEADZONE = 0.15;
const TURN_SPEED = Math.PI * 1.2;
const PT_SIZE_MIN = 0.0005;
const PT_SIZE_MAX = 0.05;
const PT_SIZE_STEP = 0.0005;
const PT_SIZE_SCALE = 3;
const CLOUD_ROTATE_SPEED = 1.4;
const CLOUD_TILT_LIMIT = Math.PI / 3;
const CLOUD_ROTATE_STEP = Math.PI / 18;
const CLOUD_TILT_STEP = Math.PI / 36;

const CLOUD_PRESETS = {
  [POINTCLOUD_PATH]: {
    label: 'Ouster Frame 001',
    pointSize: 0.004,
    cloudRotation: [-0.12, 0.35, 0],
    orbitTarget: [0, 1.2, 0],
    cameraPosition: [0, 2.1, 7.5],
  },
};

// ── State ──────────────────────────────────────────────────

let renderer, scene, camera, cameraRig, orbit, pointsMaterial, pointSprite;
let teleportMarker, teleportLine;
let cloudRoot, pointCloud, activeCloudPreset;
let useSoftPoints = DEFAULT_SOFT_POINTS;
const clock = new THREE.Clock();
const btnReady = { up: true, down: true };

// Reusable temp objects (avoids per-frame allocation)
const _dir = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _move = new THREE.Vector3();
const _hitPoint = new THREE.Vector3();
const _groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const _target = new THREE.Vector3();

// ── Bootstrap ──────────────────────────────────────────────

init();

function init() {
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(BG_COLOR);

  cloudRoot = new THREE.Group();
  scene.add(cloudRoot);

  scene.add(new THREE.GridHelper(20, 40, 0x333333, 0x1a1a1a));
  scene.add(new THREE.AxesHelper(0.3));

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 0.3);
  sun.position.set(5, 10, 5);
  scene.add(sun);

  // Camera + rig (rig is moved during VR locomotion)
  camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 1000);
  camera.position.set(0, 1.6, 3);

  cameraRig = new THREE.Group();
  cameraRig.add(camera);
  scene.add(cameraRig);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(innerWidth, innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.xr.enabled = true;
  renderer.xr.setFoveation(XR_FOVEATION);
  document.getElementById('container').appendChild(renderer.domElement);
  pointSprite = createPointSprite();

  // VR enter button
  document.body.appendChild(VRButton.createButton(renderer));

  // Desktop orbit controls
  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 1, 0);
  orbit.enableDamping = true;
  orbit.update();

  setupControllers();
  createTeleportVisuals();
  loadPointCloud(POINTCLOUD_PATH);

  // Window resize
  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // Desktop keyboard: +/- for point size
  window.addEventListener('keydown', (e) => {
    if (!pointsMaterial) return;
    if (e.key === '+' || e.key === '=') {
      pointsMaterial.size = Math.min(PT_SIZE_MAX, pointsMaterial.size + PT_SIZE_STEP);
      updateSizeUI();
    } else if (e.key === '-' || e.key === '_') {
      pointsMaterial.size = Math.max(PT_SIZE_MIN, pointsMaterial.size - PT_SIZE_STEP);
      updateSizeUI();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      rotateCloud(CLOUD_ROTATE_STEP, 0);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      rotateCloud(-CLOUD_ROTATE_STEP, 0);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      rotateCloud(0, CLOUD_TILT_STEP);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      rotateCloud(0, -CLOUD_TILT_STEP);
    } else if (e.key.toLowerCase() === 'r') {
      resetViewToPreset();
    } else if (e.key.toLowerCase() === 'v') {
      togglePointRenderMode();
    }
  });

  // Disable orbit controls during VR
  renderer.xr.addEventListener('sessionstart', () => {
    orbit.enabled = false;
    document.getElementById('info').style.display = 'none';
  });
  renderer.xr.addEventListener('sessionend', () => {
    orbit.enabled = true;
    document.getElementById('info').style.display = '';
  });

  renderer.setAnimationLoop(frame);
}

// ── VR Controllers ─────────────────────────────────────────

function setupControllers() {
  const factory = new XRControllerModelFactory();

  for (let i = 0; i < 2; i++) {
    const controller = renderer.xr.getController(i);
    cameraRig.add(controller);

    const grip = renderer.xr.getControllerGrip(i);
    grip.add(factory.createControllerModel(grip));
    cameraRig.add(grip);
  }
}

// ── Teleport Visuals ───────────────────────────────────────

function createTeleportVisuals() {
  const ringGeo = new THREE.RingGeometry(0.15, 0.2, 32).rotateX(-Math.PI / 2);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x44ffaa,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });
  teleportMarker = new THREE.Mesh(ringGeo, ringMat);
  teleportMarker.visible = false;
  scene.add(teleportMarker);

  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(),
    new THREE.Vector3(),
  ]);
  const lineMat = new THREE.LineBasicMaterial({
    color: 0x44ffaa,
    transparent: true,
    opacity: 0.4,
  });
  teleportLine = new THREE.Line(lineGeo, lineMat);
  teleportLine.visible = false;
  scene.add(teleportLine);
}

function createPointSprite() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.55, 'rgba(255,255,255,0.95)');
  gradient.addColorStop(0.82, 'rgba(255,255,255,0.35)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  return new THREE.CanvasTexture(canvas);
}

function createPointsMaterial(pointSize) {
  return new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    sizeAttenuation: true,
    alphaMap: useSoftPoints ? pointSprite : null,
    transparent: useSoftPoints,
    alphaTest: useSoftPoints ? 0.15 : 0,
    depthWrite: !useSoftPoints,
  });
}

function togglePointRenderMode() {
  if (!pointCloud || !pointsMaterial) return;

  const size = pointsMaterial.size;
  useSoftPoints = !useSoftPoints;

  const nextMaterial = createPointsMaterial(size);
  pointCloud.material.dispose();
  pointCloud.material = nextMaterial;
  pointsMaterial = nextMaterial;

  console.log(`Point render mode: ${useSoftPoints ? 'soft sprites' : 'solid squares'}`);
}

// ── Point Cloud Loading ────────────────────────────────────

function loadPointCloud(path) {
  const ext = path.split('.').pop().toLowerCase();
  activeCloudPreset = CLOUD_PRESETS[path] || {};
  document.getElementById('cloud-name').textContent =
    activeCloudPreset.label || path.split('/').pop();

  const onLoaded = (result) => {
    const geometry = result.geometry || result; // PCDLoader returns Points, PLYLoader returns BufferGeometry
    buildCloud(geometry, activeCloudPreset);
  };

  if (ext === 'ply') {
    new PLYLoader().load(path, onLoaded, onProgress, onError);
  } else if (ext === 'pcd') {
    new PCDLoader().load(path, onLoaded, onProgress, onError);
  } else {
    onError(new Error(`Unsupported format: .${ext}`));
  }
}

function onProgress(e) {
  if (e.lengthComputable) {
    const pct = (e.loaded / e.total * 100) | 0;
    document.querySelector('#loading p').textContent = `Loading… ${pct}%`;
  }
}

function onError(err) {
  console.error('Point cloud load failed:', err);
  document.querySelector('#loading p').textContent = 'Failed to load point cloud.';
  document.querySelector('#loading .spinner').style.display = 'none';
}

function buildCloud(geometry, preset = {}) {
  if (pointCloud) {
    cloudRoot.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }

  geometry.computeBoundingBox();
  const bbox = geometry.boundingBox;
  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());

  if (AUTO_CENTER) {
    geometry.translate(-center.x, -bbox.min.y, -center.z);
  }

  // Generate height-based coloring when vertex colors are missing
  if (!geometry.hasAttribute('color')) {
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);
    const yRange = size.y || 1;
    const c = new THREE.Color();
    for (let i = 0; i < positions.count; i++) {
      const t = positions.getY(i) / yRange;
      c.setHSL(0.7 - t * 0.7, 0.85, 0.5);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  }

  // Auto-scale point size relative to cloud dimensions
  const avgDim = (size.x + size.y + size.z) / 3;
  const autoSize = Math.max(PT_SIZE_MIN, Math.min(PT_SIZE_MAX, avgDim * 0.0012));
  const basePointSize = preset.pointSize ?? autoSize;
  const pointSize = Math.max(PT_SIZE_MIN, Math.min(PT_SIZE_MAX, basePointSize * PT_SIZE_SCALE));

  pointsMaterial = createPointsMaterial(pointSize);

  pointCloud = new THREE.Points(geometry, pointsMaterial);
  cloudRoot.add(pointCloud);
  applyCloudPreset(size, preset);

  // Update UI
  document.getElementById('loading').style.display = 'none';
  document.getElementById('info').style.display = '';
  document.getElementById('stats').style.display = '';
  document.getElementById('point-count').textContent =
    geometry.attributes.position.count.toLocaleString();
  updateSizeUI();

  console.log(
    `Loaded ${geometry.attributes.position.count.toLocaleString()} points — ` +
    `${size.x.toFixed(2)} × ${size.y.toFixed(2)} × ${size.z.toFixed(2)} m`
  );
}

function applyCloudPreset(size, preset = {}) {
  const defaultTarget = [0, size.y * 0.35, 0];
  const defaultCamera = [0, size.y * 0.55 + 0.6, Math.max(size.x, size.z) * 1.1];
  const rotation = preset.cloudRotation || [0, 0, 0];
  const target = preset.orbitTarget || defaultTarget;
  const cameraPosition = preset.cameraPosition || defaultCamera;

  cloudRoot.rotation.set(rotation[0] || 0, rotation[1] || 0, rotation[2] || 0);
  orbit.target.fromArray(target);
  camera.position.fromArray(cameraPosition);
  cameraRig.position.set(0, 0, 0);
  cameraRig.rotation.set(0, 0, 0);
  orbit.update();
}

// ── VR Input Handling ──────────────────────────────────────

function getXRInput() {
  const session = renderer.xr.getSession();
  if (!session) return null;

  const result = {};
  for (let i = 0; i < session.inputSources.length; i++) {
    const src = session.inputSources[i];
    if (src.handedness === 'left' || src.handedness === 'right') {
      result[src.handedness] = {
        gamepad: src.gamepad,
        controller: renderer.xr.getController(i),
      };
    }
  }
  return result;
}

function readStick(gamepad, axisIndex) {
  if (!gamepad) return 0;
  // Quest puts thumbstick on axes 2,3; other controllers may use 0,1
  const idx = gamepad.axes.length >= 4 ? axisIndex + 2 : axisIndex;
  const raw = gamepad.axes[idx] || 0;
  if (Math.abs(raw) < STICK_DEADZONE) return 0;
  return (raw - Math.sign(raw) * STICK_DEADZONE) / (1 - STICK_DEADZONE);
}

function handleVRInput(dt) {
  const input = getXRInput();
  if (!input) return;

  const left = input.left;
  const right = input.right;
  const lgp = left?.gamepad;
  const rgp = right?.gamepad;

  if (lgp?.buttons[1]?.pressed && rgp?.buttons[1]?.pressed) {
    cameraRig.position.set(0, 0, 0);
    cameraRig.rotation.set(0, 0, 0);
    return;
  }

  // ── Fly locomotion (left thumbstick) ──────────────────
  // Moves in the direction the head is facing, including vertical
  const lx = readStick(lgp, 0);
  const ly = readStick(lgp, 1);
  if (lx !== 0 || ly !== 0) {
    camera.getWorldDirection(_fwd);
    _right.crossVectors(_fwd, camera.up).normalize();
    _move.set(0, 0, 0)
      .addScaledVector(_fwd, -ly * MOVE_SPEED * dt)
      .addScaledVector(_right, lx * MOVE_SPEED * dt);
    cameraRig.position.add(_move);
  }

  const rx = readStick(rgp, 0);
  const ry = readStick(rgp, 1);
  const cloudRotateMode = !!rgp?.buttons[1]?.pressed;

  if (cloudRotateMode) {
    hideTeleport();
    rotateCloud(-rx * CLOUD_ROTATE_SPEED * dt, -ry * CLOUD_ROTATE_SPEED * dt);
  } else if (rx !== 0) {
    rotateRigAroundHead(-rx * TURN_SPEED * dt);
  }

  // ── Teleport (right trigger) ──────────────────────────
  // Hold trigger to aim, release to teleport to ground plane
  if (right && rgp && !cloudRotateMode) {
    const trigger = rgp.buttons[0];

    if (trigger?.pressed) {
      right.controller.getWorldPosition(_pos);
      right.controller.getWorldDirection(_dir);

      const ray = new THREE.Ray(_pos, _dir);
      const hit = ray.intersectPlane(_groundPlane, _hitPoint);

      if (hit && _hitPoint.distanceTo(_pos) < 30) {
        teleportMarker.position.set(_hitPoint.x, 0.01, _hitPoint.z);
        teleportMarker.visible = true;

        const attr = teleportLine.geometry.attributes.position;
        attr.setXYZ(0, _pos.x, _pos.y, _pos.z);
        attr.setXYZ(1, _hitPoint.x, 0.01, _hitPoint.z);
        attr.needsUpdate = true;
        teleportLine.visible = true;
      } else {
        hideTeleport();
      }
    } else if (!trigger?.pressed && teleportMarker.visible) {
      // Execute teleport: move rig so the user's XZ position matches the marker
      camera.getWorldPosition(_pos);
      cameraRig.position.x += teleportMarker.position.x - _pos.x;
      cameraRig.position.z += teleportMarker.position.z - _pos.z;
      hideTeleport();
    } else {
      hideTeleport();
    }
  }

  // ── Point size: A/X = bigger, B/Y = smaller ──────────
  if (pointsMaterial) {
    adjustPointSize(lgp);
    adjustPointSize(rgp);
  }

}

function adjustPointSize(gp) {
  if (!gp) return;

  // Button 4 = A (right) or X (left) → increase
  if (gp.buttons[4]?.pressed && btnReady.up) {
    pointsMaterial.size = Math.min(PT_SIZE_MAX, pointsMaterial.size + PT_SIZE_STEP);
    btnReady.up = false;
    updateSizeUI();
  }
  if (!gp.buttons[4]?.pressed) btnReady.up = true;

  // Button 5 = B (right) or Y (left) → decrease
  if (gp.buttons[5]?.pressed && btnReady.down) {
    pointsMaterial.size = Math.max(PT_SIZE_MIN, pointsMaterial.size - PT_SIZE_STEP);
    btnReady.down = false;
    updateSizeUI();
  }
  if (!gp.buttons[5]?.pressed) btnReady.down = true;
}

function hideTeleport() {
  teleportMarker.visible = false;
  teleportLine.visible = false;
}

function rotateRigAroundHead(angle) {
  camera.getWorldPosition(_pos);
  cameraRig.position.sub(_pos);
  cameraRig.position.applyAxisAngle(THREE.Object3D.DEFAULT_UP, angle);
  cameraRig.position.add(_pos);
  cameraRig.rotateY(angle);
}

function rotateCloud(yawDelta, pitchDelta) {
  if (!pointCloud) return;
  cloudRoot.rotation.y += yawDelta;
  cloudRoot.rotation.x = THREE.MathUtils.clamp(
    cloudRoot.rotation.x + pitchDelta,
    -CLOUD_TILT_LIMIT,
    CLOUD_TILT_LIMIT
  );
}

function resetViewToPreset() {
  if (!pointCloud) return;

  if (activeCloudPreset?.pointSize !== undefined) {
    pointsMaterial.size = THREE.MathUtils.clamp(activeCloudPreset.pointSize, PT_SIZE_MIN, PT_SIZE_MAX);
  }

  pointCloud.geometry.computeBoundingBox();
  const bbox = pointCloud.geometry.boundingBox;
  const size = bbox.getSize(_target);
  applyCloudPreset(size, activeCloudPreset);
  updateSizeUI();
}

function updateSizeUI() {
  if (pointsMaterial) {
    document.getElementById('ps-val').textContent = pointsMaterial.size.toFixed(3);
  }
}

// ── Render Loop ────────────────────────────────────────────

function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);

  if (renderer.xr.isPresenting) {
    handleVRInput(dt);
  } else {
    orbit.update();
  }

  renderer.render(scene, camera);
}
