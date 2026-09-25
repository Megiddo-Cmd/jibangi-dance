import { FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const FATTY_MODEL_PATH = './fatty.glb';

const $ = (selector) => document.querySelector(selector);
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (from, to, amount) => from + (to - from) * amount;
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const midpoint = (a, b) => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
  visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
});
const angleBetween = (a, b) => Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;

const elements = {
  camera: $('#camera'),
  stage: $('#stage'),
  rig: $('#rig'),
  fattyRig: null,
  bodyPart: null,
  headPart: null,
  noPose: $('#noPoseIllustration'),
  startButton: $('#startButton'),
  stopButton: $('#stopButton'),
  mirrorButton: $('#mirrorButton'),
  danceExitButton: $('#danceExitButton'),
  cameraSelect: $('#cameraSelect'),
  cameraHint: $('#cameraHint'),
  refreshCameraButton: $('#refreshCameraButton'),
  smoothingRange: $('#smoothingRange'),
  smoothingValue: $('#smoothingValue'),
  stageMessage: $('#stageMessage'),
  stageMessageText: $('#stageMessageText'),
  livePill: $('#livePill'),
  liveLabel: $('#liveLabel'),
  fpsValue: $('#fpsValue'),
  confidenceValue: $('#confidenceValue'),
};

let fattyScene;
let fattyCamera;
let fattyRenderer;
let fattyModel;
let fattyBones = {};
let fattyModelReady = false;
let fattyRenderWidth = 0;
let fattyRenderHeight = 0;
const fattyRestWorld = {};
// fatty.glb의 실제 스켈레톤 이름입니다.
const fattyBoneNames = [
  'body', 'head',
  'upperArm.L', 'forearm.L',
  'upperArm.R', 'forearm.R',
  'thigh.L', 'shin.L',
  'thigh.R', 'shin.R',
];

// 앱에서 사용하는 의미 있는 이름 → GLB의 실제 본 이름
const fattyBoneMap = {
  Body: 'body',
  Head: 'head',
  LeftUpperArm: 'upperArm.L',
  LeftForearm: 'forearm.L',
  RightUpperArm: 'upperArm.R',
  RightForearm: 'forearm.R',
  LeftUpperLeg: 'thigh.L',
  LeftLowerLeg: 'shin.L',
  RightUpperLeg: 'thigh.R',
  RightLowerLeg: 'shin.R',
};

const landmarkIndex = {
  nose: 0,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
};

let poseLandmarker;
let cameraStream;
let animationFrame;
let isRunning = false;
let isMirrored = true;
let filteredLandmarks = null;
let lastPoseTime = 0;
let lastFrameAt = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();

function hasCameraApi() {
  return Boolean(navigator.mediaDevices?.getUserMedia && navigator.mediaDevices?.enumerateDevices);
}

function getCameraErrorMessage(error) {
  if (!window.isSecureContext) {
    return '모바일에서는 HTTPS 주소로 열어야 카메라를 쓸 수 있어요. 맥의 localhost 주소는 아이폰에서 사용할 수 없어요.';
  }
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
    return '브라우저의 카메라 권한을 허용한 뒤 다시 눌러 주세요.';
  }
  if (error?.name === 'NotFoundError' || error?.name === 'OverconstrainedError') {
    return '선택한 카메라를 찾지 못했어요. 카메라 목록에서 다른 카메라를 골라 주세요.';
  }
  if (error?.name === 'NotReadableError') {
    return '카메라가 다른 앱에서 사용 중이에요. FaceTime이나 다른 카메라 앱을 닫아 주세요.';
  }
  return '카메라를 시작하지 못했어요. 연결 상태와 권한을 확인해 주세요.';
}

async function refreshCameraList(selectedDeviceId = elements.cameraSelect.value) {
  if (!window.isSecureContext) {
    elements.cameraSelect.disabled = true;
    elements.refreshCameraButton.disabled = false;
    elements.cameraHint.textContent = '모바일은 HTTPS 주소에서 열어야 카메라를 사용할 수 있어요.';
    return;
  }
  if (!hasCameraApi()) {
    elements.cameraSelect.disabled = true;
    elements.refreshCameraButton.disabled = true;
    elements.cameraHint.textContent = '이 브라우저에서는 카메라 기능을 사용할 수 없어요.';
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === 'videoinput');
    const currentValue = selectedDeviceId || elements.cameraSelect.value;
    elements.cameraSelect.replaceChildren(new Option('자동으로 선택', ''));

    cameras.forEach((device, index) => {
      const fallbackLabel = `카메라 ${index + 1}`;
      const readableLabel = device.label || fallbackLabel;
      const label = /iphone|continuity/i.test(readableLabel) ? `📱 ${readableLabel}` : readableLabel;
      elements.cameraSelect.append(new Option(label, device.deviceId));
    });

    if (currentValue && cameras.some((camera) => camera.deviceId === currentValue)) {
      elements.cameraSelect.value = currentValue;
    }
    elements.cameraSelect.disabled = cameras.length === 0;
    elements.refreshCameraButton.disabled = false;
  } catch (error) {
    console.warn('카메라 목록을 불러오지 못했어요.', error);
  }
}

async function requestCameraPermissionAndRefresh() {
  if (!hasCameraApi() || !window.isSecureContext) {
    await refreshCameraList();
    return;
  }
  try {
    const previewStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'user' } },
      audio: false,
    });
    await refreshCameraList();
    previewStream.getTracks().forEach((track) => track.stop());
    elements.cameraHint.textContent = '목록에서 iPhone 또는 원하는 카메라를 골라 주세요.';
  } catch (error) {
    elements.cameraHint.textContent = getCameraErrorMessage(error);
  }
}

function setAppState(state, label) {
  elements.livePill.dataset.state = state;
  elements.liveLabel.textContent = label;
}

function showMessage(message, visible = true) {
  elements.stageMessageText.textContent = message;
  elements.stageMessage.classList.toggle('is-hidden', !visible);
}

function setRigVisible(visible) {
  if (elements.rig) elements.rig.style.opacity = visible ? '1' : '0';
  if (elements.noPose) elements.noPose.style.opacity = visible ? '0' : '0.92';
}

function mirroredLandmarks(rawLandmarks) {
  return rawLandmarks.map((point) => ({
    x: isMirrored ? 1 - point.x : point.x,
    y: point.y,
    z: point.z ?? 0,
    visibility: point.visibility ?? 1,
  }));
}

function smoothLandmarks(rawLandmarks) {
  const smoothing = Number(elements.smoothingRange.value) / 100;
  const follow = 0.1 + (1 - smoothing) * 0.82;
  const next = mirroredLandmarks(rawLandmarks);

  if (!filteredLandmarks) {
    filteredLandmarks = next;
    return filteredLandmarks;
  }

  filteredLandmarks = filteredLandmarks.map((previous, index) => {
    const target = next[index];
    const isReliable = target.visibility >= 0.42;
    return {
      x: isReliable ? lerp(previous.x, target.x, follow) : previous.x,
      y: isReliable ? lerp(previous.y, target.y, follow) : previous.y,
      z: isReliable ? lerp(previous.z, target.z, follow) : previous.z,
      visibility: target.visibility,
    };
  });
  return filteredLandmarks;
}

function getPoint(landmarks, name) {
  return landmarks[landmarkIndex[name]];
}

function setSegment(id, start, end, width = 100) {
  const node = document.getElementById(id);
  const x1 = start.x * 1000;
  const y1 = start.y * 562.5;
  const x2 = end.x * 1000;
  const y2 = end.y * 562.5;
  const length = Math.max(1, Math.hypot(x2 - x1, y2 - y1));
  const scaleX = length / width;
  const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
  node.setAttribute('transform', `translate(${x1} ${y1}) rotate(${angle}) scale(${scaleX} 1)`);
}

function setPoint(id, point, rotation = 0, scale = 1) {
  const node = document.getElementById(id);
  node.setAttribute('transform', `translate(${point.x * 1000} ${point.y * 562.5}) rotate(${rotation}) scale(${scale})`);
}

function pointAlong(point, angle, length) {
  const radians = angle * Math.PI / 180;
  return {
    x: point.x + Math.cos(radians) * length / 1000,
    y: point.y + Math.sin(radians) * length / 562.5,
  };
}

function rotatedOffset(origin, offsetX, offsetY, rotation) {
  const radians = rotation * Math.PI / 180;
  return {
    x: origin.x + (offsetX * Math.cos(radians) - offsetY * Math.sin(radians)) / 1000,
    y: origin.y + (offsetX * Math.sin(radians) + offsetY * Math.cos(radians)) / 562.5,
  };
}

function boneLength(start, end) {
  return Math.max(1, Math.hypot((end.x - start.x) * 1000, (end.y - start.y) * 562.5));
}

function setAdaptiveSegment(id, anchor, sourceStart, sourceEnd, minLength, maxLength) {
  const angle = angleBetween(sourceStart, sourceEnd);
  const length = clamp(boneLength(sourceStart, sourceEnd), minLength, maxLength);
  const end = pointAlong(anchor, angle, length);
  setSegment(id, anchor, end);
  return end;
}

function visualLineAngle(first, second) {
  const visualLeft = first.x <= second.x ? first : second;
  const visualRight = visualLeft === first ? second : first;
  return angleBetween(visualLeft, visualRight);
}

function normalizeAngle(angle) {
  while (angle > Math.PI) angle -= Math.PI * 2;
  while (angle < -Math.PI) angle += Math.PI * 2;
  return angle;
}

function captureFattyRestPose() {
  if (!fattyModel) return;

  fattyModel.updateMatrixWorld(true);

  for (const name of fattyBoneNames) {
    const bone = fattyBones[name];
    if (!bone) continue;

    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();

    bone.getWorldPosition(worldPos);
    bone.getWorldQuaternion(worldQuat);

    // 실제 자식 본 방향을 rest 방향으로 사용
    const child = bone.children.find((node) => node.isBone);

    let axis = new THREE.Vector3(0, 1, 0);

    if (child) {
      const childPos = new THREE.Vector3();
      child.getWorldPosition(childPos);

      axis.copy(childPos).sub(worldPos);

      if (axis.lengthSq() > 0.000001) {
        axis.normalize();
      } else {
        axis.set(0, 1, 0);
      }
    } else {
      // 끝 본은 자신의 로컬 +Y 방향을 사용
      axis.set(0, 1, 0)
        .applyQuaternion(worldQuat)
        .normalize();
    }

    fattyRestWorld[name] = {
      worldPos: worldPos.clone(),
      worldQuat: worldQuat.clone(),
      axis: axis.clone(),
    };
  }
}

function aimBoneToScreenSegment(name, first, second) {
  const actualName = fattyBoneMap[name] ?? name;
  const bone = fattyBones[actualName];
  const rest = fattyRestWorld[actualName];

  if (!bone || !rest) return;

  // MediaPipe: Y 아래쪽
  // Three.js: Y 위쪽
  const target = new THREE.Vector3(
    second.x - first.x,
    -(second.y - first.y),
    0
  );

  if (target.lengthSq() < 0.000001) return;

  target.normalize();

  // REST 본 방향 → 현재 MediaPipe 관절 방향
  const correction = new THREE.Quaternion()
    .setFromUnitVectors(rest.axis, target);

  const desiredWorld = correction
    .clone()
    .multiply(rest.worldQuat);

  // 부모 본의 월드 회전을 제거해서
  // 현재 본에는 LOCAL 회전만 적용
  if (bone.parent && bone.parent.isBone) {
    const parentWorld = new THREE.Quaternion();

    bone.parent.getWorldQuaternion(parentWorld);

    bone.quaternion.copy(
      parentWorld
        .invert()
        .multiply(desiredWorld)
    );
  } else {
    const modelWorld = new THREE.Quaternion();

    fattyModel.getWorldQuaternion(modelWorld);

    bone.quaternion.copy(
      modelWorld
        .invert()
        .multiply(desiredWorld)
    );
  }
}

function aimHeadToPose(name, neck, nose) {
  aimBoneToScreenSegment(name, neck, nose);
}

function updateRig(landmarks) {
  if (!fattyModelReady) return;

  const leftShoulder = getPoint(landmarks, 'leftShoulder');
  const rightShoulder = getPoint(landmarks, 'rightShoulder');
  const leftElbow = getPoint(landmarks, 'leftElbow');
  const rightElbow = getPoint(landmarks, 'rightElbow');
  const leftWrist = getPoint(landmarks, 'leftWrist');
  const rightWrist = getPoint(landmarks, 'rightWrist');
  const leftHip = getPoint(landmarks, 'leftHip');
  const rightHip = getPoint(landmarks, 'rightHip');
  const leftKnee = getPoint(landmarks, 'leftKnee');
  const rightKnee = getPoint(landmarks, 'rightKnee');
  const leftAnkle = getPoint(landmarks, 'leftAnkle');
  const rightAnkle = getPoint(landmarks, 'rightAnkle');
  const nose = getPoint(landmarks, 'nose');

  const shoulderCenter = midpoint(leftShoulder, rightShoulder);
  const hipCenter = midpoint(leftHip, rightHip);

  // 모델의 위치는 골반 중심만 따라갑니다. 메시/본 길이/스케일은 매 프레임 건드리지 않습니다.
  const aspect = fattyRenderHeight ? fattyRenderWidth / fattyRenderHeight : 16 / 9;
  const worldHeight = 2.6;
  fattyModel.scale.setScalar(1);
  fattyModel.position.x = (0.5 - hipCenter.x) * worldHeight * aspect;
  fattyModel.position.y = (0.5 - hipCenter.y) * worldHeight;
  fattyModel.updateMatrixWorld(true);

  // 핵심: MediaPipe 관절 ↔ GLB 본을 정확히 1:1로 대응시킵니다.
  // Body: 골반 → 어깨
  aimBoneToScreenSegment('Body', hipCenter, shoulderCenter);
  // Head: 어깨 중앙 → 코
  aimHeadToPose('Head', shoulderCenter, nose);

  // 왼팔
  aimBoneToScreenSegment('LeftUpperArm', leftShoulder, leftElbow);
  aimBoneToScreenSegment('LeftForearm', leftElbow, leftWrist);
  // 오른팔
  aimBoneToScreenSegment('RightUpperArm', rightShoulder, rightElbow);
  aimBoneToScreenSegment('RightForearm', rightElbow, rightWrist);

  // 왼다리
  aimBoneToScreenSegment('LeftUpperLeg', leftHip, leftKnee);
  aimBoneToScreenSegment('LeftLowerLeg', leftKnee, leftAnkle);
  // 오른다리
  aimBoneToScreenSegment('RightUpperLeg', rightHip, rightKnee);
  aimBoneToScreenSegment('RightLowerLeg', rightKnee, rightAnkle);

  // 손/발은 해당 마지막 관절에 붙어 있도록 두고, 별도 스케일/늘이기는 하지 않습니다.
  if (fattyRenderer) fattyRenderer.render(fattyScene, fattyCamera);
}

function updateStats(confidence, now) {
  frameCount += 1;
  if (now - lastFpsUpdate < 500) return;
  const fps = Math.round(frameCount * 1000 / (now - lastFpsUpdate));
  elements.fpsValue.textContent = String(clamp(fps, 0, 60));
  elements.confidenceValue.textContent = `${Math.round(confidence * 100)}%`;
  frameCount = 0;
  lastFpsUpdate = now;
}

async function loadFattyModel() {
  if (fattyModelReady) return;

  fattyScene = new THREE.Scene();
  fattyCamera = new THREE.OrthographicCamera(-2, 2, 1.3, -1.3, 0.01, 100);
  fattyCamera.position.set(0, 0, 8);
  fattyCamera.lookAt(0, 0, 0);

  fattyRenderer = new THREE.WebGLRenderer({
    canvas: elements.rig,
    alpha: true,
    antialias: true,
    powerPreference: 'high-performance',
  });
  fattyRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  fattyRenderer.outputColorSpace = THREE.SRGBColorSpace;
  fattyRenderer.setClearColor(0x000000, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 2.2);
  fattyScene.add(ambient);
  const key = new THREE.DirectionalLight(0xffffff, 2.8);
  key.position.set(1.5, 2.5, 4);
  fattyScene.add(key);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(FATTY_MODEL_PATH);
  fattyModel = gltf.scene;
  fattyModel.traverse((object) => {
    if (object.isBone && fattyBoneNames.includes(object.name)) {
      fattyBones[object.name] = object;
    }
    if (object.isMesh) {
      object.frustumCulled = false;
      object.castShadow = false;
      object.receiveShadow = false;
      if (object.material) {
        object.material.roughness = Math.min(object.material.roughness ?? 0.7, 0.75);
      }
    }
  });

  // GLB의 기본 자세는 몸통이 Z축을 향하고 있어 화면에서 눕습니다.
  // X축으로 90도 세워서 MediaPipe의 2D X/Y 방향과 자연스럽게 맞춥니다.
  fattyModel.rotation.set(Math.PI / 2, 0, 0);
  fattyModel.scale.setScalar(1);

  fattyScene.add(fattyModel);
  fattyModel.updateMatrixWorld(true);
  captureFattyRestPose();
  fattyModelReady = true;
  resizeFattyRenderer();
  setRigVisible(false);
  fattyRenderer.render(fattyScene, fattyCamera);
}

function resizeFattyRenderer() {
  if (!fattyRenderer || !elements.rig) return;
  const rect = elements.rig.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  fattyRenderWidth = rect.width;
  fattyRenderHeight = rect.height;
  fattyRenderer.setSize(rect.width, rect.height, false);
  const aspect = rect.width / rect.height;
  const height = 2.6;
  fattyCamera.left = -(height * aspect) / 2;
  fattyCamera.right = (height * aspect) / 2;
  fattyCamera.top = height / 2;
  fattyCamera.bottom = -height / 2;
  fattyCamera.updateProjectionMatrix();
  if (fattyModel) fattyRenderer.render(fattyScene, fattyCamera);
}

async function createLandmarker() {
  if (poseLandmarker) return poseLandmarker;
  setAppState('loading', '모델 준비 중');
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  const sharedOptions = {
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      ...sharedOptions,
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
    });
  } catch (gpuError) {
    console.warn('GPU 가속을 사용할 수 없어 CPU 모드로 전환해요.', gpuError);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      ...sharedOptions,
      baseOptions: { modelAssetPath: MODEL_PATH },
    });
  }
  return poseLandmarker;
}

async function startCamera() {
  if (isRunning) return;
  if (!hasCameraApi()) {
    showMessage(getCameraErrorMessage(), true);
    return;
  }
  elements.startButton.disabled = true;
  showMessage('지방이를 깨우는 중…', true);
  setAppState('loading', '준비 중');

  try {
    await createLandmarker();
    const selectedDeviceId = elements.cameraSelect.value;
    const videoConstraints = {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      ...(selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : { facingMode: { ideal: 'user' } }),
    };
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false,
    });
    const activeDeviceId = cameraStream.getVideoTracks()[0]?.getSettings().deviceId;
    await refreshCameraList(activeDeviceId || selectedDeviceId);
    elements.camera.srcObject = cameraStream;
    await elements.camera.play();
    isRunning = true;
    document.body.classList.add('dance-mode');
    elements.stopButton.disabled = false;
    setAppState('live', '라이브');
    showMessage('화면 안에 전신을 맞춰 주세요', true);
    lastFrameAt = performance.now();
    lastFpsUpdate = lastFrameAt;
    frameCount = 0;
    animationFrame = requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    document.body.classList.remove('dance-mode');
    elements.startButton.disabled = false;
    elements.stopButton.disabled = true;
    setAppState('idle', '대기 중');
    showMessage(getCameraErrorMessage(error), true);
  }
}

function stopCamera() {
  isRunning = false;
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  document.body.classList.remove('dance-mode');
  filteredLandmarks = null;
  lastPoseTime = 0;
  elements.camera.srcObject = null;
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  elements.fpsValue.textContent = '--';
  elements.confidenceValue.textContent = '--';
  setAppState('idle', '대기 중');
  showMessage('아래 버튼을 눌러 카메라를 시작해 주세요', true);
  setRigVisible(false);
}

function processFrame(now) {
  if (!isRunning) return;
  animationFrame = requestAnimationFrame(processFrame);
  if (elements.camera.readyState < 2 || now - lastFrameAt < 30) return;
  lastFrameAt = now;

  const result = poseLandmarker.detectForVideo(elements.camera, now);
  const rawLandmarks = result.landmarks?.[0];
  if (!rawLandmarks) {
    if (now - lastPoseTime > 450) {
      setRigVisible(false);
      elements.confidenceValue.textContent = '--';
      showMessage('몸 전체가 보이도록 조금만 뒤로 가 주세요', true);
    }
    return;
  }

  const landmarks = smoothLandmarks(rawLandmarks);
  const trackedNames = ['leftShoulder', 'rightShoulder', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];
  const confidence = trackedNames.reduce((sum, name) => sum + getPoint(landmarks, name).visibility, 0) / trackedNames.length;
  if (confidence < 0.2) return;
  lastPoseTime = now;
  updateRig(landmarks);
  updateStats(confidence, now);
  setRigVisible(true);
  showMessage('지방이가 따라 추는 중이에요 ✨', false);
}

function toggleMirror() {
  isMirrored = !isMirrored;
  elements.camera.classList.toggle('is-mirrored', isMirrored);
  elements.mirrorButton.setAttribute('aria-pressed', String(isMirrored));
}

elements.startButton.addEventListener('click', startCamera);
elements.stopButton.addEventListener('click', stopCamera);
elements.danceExitButton.addEventListener('click', stopCamera);
elements.mirrorButton.addEventListener('click', toggleMirror);
elements.cameraSelect.addEventListener('change', async () => {
  if (!isRunning) return;
  stopCamera();
  await startCamera();
});
elements.refreshCameraButton.addEventListener('click', requestCameraPermissionAndRefresh);
elements.smoothingRange.addEventListener('input', (event) => {
  elements.smoothingValue.textContent = `${event.target.value}%`;
});

elements.camera.classList.toggle('is-mirrored', isMirrored);
document.body.classList.remove('dance-mode');
setRigVisible(false);
loadFattyModel().catch((error) => {
  console.error('지방이 3D 모델을 불러오지 못했어요.', error);
});
window.addEventListener('resize', resizeFattyRenderer);
refreshCameraList();
if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener?.('devicechange', () => refreshCameraList());
}

window.addEventListener('beforeunload', stopCamera);
