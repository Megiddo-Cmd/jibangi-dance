import { FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/loaders/GLTFLoader.js';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const GLB_PATH = './assets/fatty_motion_capture_rigged.glb';

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const angle = (a, b) => Math.atan2(b.y - a.y, b.x - a.x);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const el = {
  camera: $('#camera'), canvas: $('#threeCanvas'), stage: $('#stage'),
  start: $('#startButton'), stop: $('#stopButton'), mirror: $('#mirrorButton'),
  cameraSelect: $('#cameraSelect'), refresh: $('#refreshCameraButton'), cameraHint: $('#cameraHint'),
  smoothing: $('#smoothingRange'), smoothingValue: $('#smoothingValue'),
  livePill: $('#livePill'), liveLabel: $('#liveLabel'), fps: $('#fpsValue'), confidence: $('#confidenceValue'),
  message: $('#stageMessage'), messageText: $('#stageMessageText'),
};

const idx = { nose:0, leftShoulder:11, rightShoulder:12, leftElbow:13, rightElbow:14, leftWrist:15, rightWrist:16, leftHip:23, rightHip:24, leftKnee:25, rightKnee:26, leftAnkle:27, rightAnkle:28 };

let landmarker = null, stream = null, raf = 0, running = false, mirrored = true;
let filtered = null, lastTime = performance.now(), frames = 0, fpsClock = performance.now();

// ---------- Three.js ----------
const scene = new THREE.Scene();
const camera3d = new THREE.OrthographicCamera(-1.5, 1.5, 1.1, -1.1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ canvas: el.canvas, alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setAnimationLoop(() => renderer.render(scene, camera3d));

scene.add(new THREE.AmbientLight(0xffffff, 2.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.2); keyLight.position.set(-2, 3, 4); scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xa98bff, 1.0); fillLight.position.set(3, 1, 2); scene.add(fillLight);

let fatty = null;
const bones = {};
const rest = {};
const poseScale = 1.55;

function resize3D() {
  const r = el.canvas.getBoundingClientRect();
  const aspect = Math.max(0.5, r.width / Math.max(1, r.height));
  const h = 1.18, w = h * aspect;
  camera3d.left = -w; camera3d.right = w; camera3d.top = h; camera3d.bottom = -h; camera3d.updateProjectionMatrix();
  renderer.setSize(r.width, r.height, false);
}
window.addEventListener('resize', resize3D);

function findRigBones(root) {
  const names = ['Fatty_Root','Body','Head','LeftUpperArm','LeftForearm','LeftHand','RightUpperArm','RightForearm','RightHand','LeftUpperLeg','LeftLowerLeg','LeftFoot','RightUpperLeg','RightLowerLeg','RightFoot'];
  names.forEach(n => bones[n] = root.getObjectByName(n));
  for (const [name, bone] of Object.entries(bones)) {
    if (bone) rest[name] = bone.rotation.z;
  }
}

new GLTFLoader().load(GLB_PATH, (gltf) => {
  fatty = gltf.scene;
  findRigBones(fatty);
  fatty.scale.setScalar(0.95);
  fatty.position.set(0, 0.03, 0);
  scene.add(fatty);
  resize3D();
}, undefined, (err) => {
  console.error(err);
  showMessage('GLB를 불러오지 못했어요. assets 폴더를 확인해 주세요.');
});

function resetRig() {
  Object.entries(rest).forEach(([name, z]) => { if (bones[name]) bones[name].rotation.z = z; });
  if (fatty) fatty.rotation.set(0, 0, 0);
}

// MediaPipe y increases downward. Three.js y increases upward.
function mpToRig(p) { return { x: (p.x - 0.5) * poseScale, y: (0.5 - p.y) * poseScale, visibility: p.visibility ?? 1 }; }

function setBoneAngle(name, start, end, restAngle = 0, strength = 1) {
  const b = bones[name];
  if (!b || !start || !end) return;
  const target = angle(start, end);
  let delta = target - restAngle;
  delta = Math.atan2(Math.sin(delta), Math.cos(delta));
  b.rotation.z = lerp(b.rotation.z, rest[name] + delta * strength, 0.55);
}

function updateRig(points) {
  if (!fatty) return;
  const p = {};
  for (const [name, i] of Object.entries(idx)) p[name] = mpToRig(points[i]);

  const ls = p.leftShoulder, rs = p.rightShoulder;
  const lh = p.leftHip, rh = p.rightHip;
  if (!ls || !rs || !lh || !rh) return;

  const shoulderMid = { x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2 };
  const hipMid = { x:(lh.x+rh.x)/2, y:(lh.y+rh.y)/2 };

  // Move/scale the puppet to the tracked person's torso.
  const torsoH = clamp(dist(shoulderMid, hipMid), 0.35, 1.2);
  fatty.position.x = lerp(fatty.position.x, shoulderMid.x * 0.18, 0.22);
  fatty.position.y = lerp(fatty.position.y, hipMid.y * 0.12, 0.22);
  const targetScale = clamp(torsoH * 2.2, 0.72, 1.45);
  const s = lerp(fatty.scale.x, targetScale, 0.12);
  fatty.scale.setScalar(s);

  // Torso tilt.
  if (bones.Body) {
    const torsoAngle = angle(hipMid, shoulderMid);
    bones.Body.rotation.z = lerp(bones.Body.rotation.z, torsoAngle - Math.PI/2, 0.35);
  }

  // Arms: generated rig's rest vectors are approximately diagonal.
  setBoneAngle('LeftUpperArm', ls, p.leftElbow, Math.atan2(0.15, -0.30));
  setBoneAngle('LeftForearm', p.leftElbow, p.leftWrist, Math.atan2(0.05, -0.18));
  setBoneAngle('RightUpperArm', rs, p.rightElbow, Math.atan2(0.10, 0.30));
  setBoneAngle('RightForearm', p.rightElbow, p.rightWrist, Math.atan2(0.05, 0.18));

  // Legs.
  setBoneAngle('LeftUpperLeg', lh, p.leftKnee, Math.PI/2);
  setBoneAngle('LeftLowerLeg', p.leftKnee, p.leftAnkle, Math.PI/2);
  setBoneAngle('RightUpperLeg', rh, p.rightKnee, Math.PI/2);
  setBoneAngle('RightLowerLeg', p.rightKnee, p.rightAnkle, Math.PI/2);

  // Head follows nose direction / torso, with a gentle yaw illusion.
  if (bones.Head && p.nose) {
    const headTilt = angle(shoulderMid, p.nose) - Math.PI/2;
    bones.Head.rotation.z = lerp(bones.Head.rotation.z, headTilt, 0.22);
  }

  // Hands and feet get a little extra follow.
  if (bones.LeftHand) bones.LeftHand.rotation.z = lerp(bones.LeftHand.rotation.z, bones.LeftForearm?.rotation.z ?? 0, 0.25);
  if (bones.RightHand) bones.RightHand.rotation.z = lerp(bones.RightHand.rotation.z, bones.RightForearm?.rotation.z ?? 0, 0.25);
  if (bones.LeftFoot) bones.LeftFoot.rotation.z = lerp(bones.LeftFoot.rotation.z, bones.LeftLowerLeg?.rotation.z ?? 0, 0.25);
  if (bones.RightFoot) bones.RightFoot.rotation.z = lerp(bones.RightFoot.rotation.z, bones.RightLowerLeg?.rotation.z ?? 0, 0.25);
}

// ---------- MediaPipe ----------
async function initPose() {
  if (landmarker) return;
  setState('loading', 'AI 준비 중');
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  });
}

function mirroredPoints(points) {
  return points.map(p => ({ x: mirrored ? 1-p.x : p.x, y:p.y, z:p.z ?? 0, visibility:p.visibility ?? 1 }));
}
function smooth(points) {
  const next = mirroredPoints(points);
  const amount = 0.1 + (1 - Number(el.smoothing.value)/100) * 0.82;
  if (!filtered) { filtered = next; return filtered; }
  filtered = filtered.map((old,i) => {
    const n = next[i];
    return { x:n.visibility > .42 ? lerp(old.x,n.x,amount) : old.x, y:n.visibility > .42 ? lerp(old.y,n.y,amount) : old.y, z:n.z, visibility:n.visibility };
  });
  return filtered;
}

function setState(state, label) { el.livePill.dataset.state = state; el.liveLabel.textContent = label; }
function showMessage(msg, visible=true) { el.messageText.textContent = msg; el.message.classList.toggle('is-hidden', !visible); }

async function refreshCameras(selected='') {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d=>d.kind==='videoinput');
    el.cameraSelect.replaceChildren(new Option('자동으로 선택',''));
    cams.forEach((d,i)=>el.cameraSelect.append(new Option(d.label || `카메라 ${i+1}`, d.deviceId)));
    if (selected && cams.some(d=>d.deviceId===selected)) el.cameraSelect.value=selected;
  } catch(e) { console.warn(e); }
}

async function startCamera() {
  try {
    await initPose();
    if (stream) stopCamera();
    const deviceId = el.cameraSelect.value;
    const constraints = { video: deviceId ? {deviceId:{exact:deviceId}, width:{ideal:1280}, height:{ideal:720}} : {facingMode:'user', width:{ideal:1280}, height:{ideal:720}}, audio:false };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    el.camera.srcObject = stream;
    await el.camera.play();
    await refreshCameras(deviceId);
    running = true; filtered = null; setState('live','모션 캡처 중');
    el.start.disabled=true; el.stop.disabled=false;
    showMessage('카메라 앞에서 움직여 보세요 ✨', true);
    cancelAnimationFrame(raf); raf=requestAnimationFrame(loop);
  } catch(e) {
    console.error(e);
    setState('idle','대기 중');
    showMessage(e.name==='NotAllowedError' ? '카메라 권한을 허용해 주세요.' : '카메라를 시작하지 못했어요. HTTPS인지 확인해 주세요.');
  }
}

function stopCamera() {
  running=false; cancelAnimationFrame(raf);
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream=null; }
  el.camera.srcObject=null; el.start.disabled=false; el.stop.disabled=true; setState('idle','대기 중');
  resetRig(); showMessage('시작 버튼을 누르면 모션 캡처가 시작돼요.');
}

function loop(now) {
  if (!running) return;
  if (el.camera.readyState >= 2 && landmarker) {
    const result = landmarker.detectForVideo(el.camera, now);
    if (result.landmarks?.length) {
      const points = smooth(result.landmarks[0]);
      updateRig(points);
      const visible = Object.values(idx).map(i=>points[i]?.visibility ?? 0);
      el.confidence.textContent = `${Math.round(visible.reduce((a,b)=>a+b,0)/visible.length*100)}%`;
      showMessage('', false);
    } else {
      el.confidence.textContent='--'; showMessage('사람을 찾는 중…', true);
    }
  }
  frames++;
  if (now-fpsClock > 700) { el.fps.textContent=Math.round(frames*1000/(now-fpsClock)); frames=0; fpsClock=now; }
  raf=requestAnimationFrame(loop);
}

el.start.addEventListener('click', startCamera);
el.stop.addEventListener('click', stopCamera);
el.mirror.addEventListener('click', ()=>{ mirrored=!mirrored; el.mirror.setAttribute('aria-pressed', String(mirrored)); });
el.refresh.addEventListener('click', ()=>refreshCameras(el.cameraSelect.value));
el.smoothing.addEventListener('input', ()=>el.smoothingValue.textContent=`${el.smoothing.value}%`);
window.addEventListener('beforeunload', stopCamera);

el.canvas.addEventListener('pointerdown', () => { if (fatty) fatty.rotation.y += 0.08; });

refreshCameras();
resize3D();
showMessage('시작 버튼을 누르면 모션 캡처가 시작돼요.');
