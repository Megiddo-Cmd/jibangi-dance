import { FilesetResolver, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm';

const WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm';
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

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
  fattyRig: $('#fattyRig'),
  bodyPart: $('#bodyPart'),
  headPart: $('#headPart'),
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
  elements.fattyRig.style.opacity = visible ? '1' : '0';
  elements.noPose.style.opacity = visible ? '0' : '0.92';
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

function updateRig(landmarks) {
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

  // 부위마다 사람의 관절을 따로 계산하고, 각 지방이 파츠를 그 위에 붙여요.
  const shoulderCenter = midpoint(leftShoulder, rightShoulder);
  const hipCenter = midpoint(leftHip, rightHip);
  const torsoAngle = visualLineAngle(leftShoulder, rightShoulder);
  const shoulderWidth = distance(leftShoulder, rightShoulder) * 1000;
  const hipWidth = distance(leftHip, rightHip) * 1000;
  const torsoLength = distance(shoulderCenter, hipCenter) * 562.5;
  const bodyWidth = clamp(Math.max(shoulderWidth * 1.55, hipWidth * 1.45), 180, 360);
  const bodyHeight = clamp(torsoLength * 1.32, 170, 280);
  const bodyCenter = {
    x: hipCenter.x,
    y: lerp(shoulderCenter.y, hipCenter.y, 0.58),
  };

  elements.bodyPart.setAttribute(
    'transform',
    `translate(${bodyCenter.x * 1000} ${bodyCenter.y * 562.5}) rotate(${torsoAngle}) scale(${bodyWidth / 164} ${bodyHeight / 127})`,
  );

  // 몸통의 어깨·골반 앵커에 각 부위를 붙이고, 사람의 방향과 길이만 따로 반영해요.
  const leftSide = isMirrored ? -1 : 1;
  const rightSide = -leftSide;
  const leftShoulderAnchor = rotatedOffset(bodyCenter, leftSide * bodyWidth * 0.32, -bodyHeight * 0.44, torsoAngle);
  const rightShoulderAnchor = rotatedOffset(bodyCenter, rightSide * bodyWidth * 0.32, -bodyHeight * 0.44, torsoAngle);
  const leftHipAnchor = rotatedOffset(bodyCenter, leftSide * bodyWidth * 0.25, bodyHeight * 0.32, torsoAngle);
  const rightHipAnchor = rotatedOffset(bodyCenter, rightSide * bodyWidth * 0.25, bodyHeight * 0.32, torsoAngle);

  const leftElbowAnchor = setAdaptiveSegment('leftUpperArm', leftShoulderAnchor, leftShoulder, leftElbow, 70, 155);
  const leftHandAnchor = setAdaptiveSegment('leftLowerArm', leftElbowAnchor, leftElbow, leftWrist, 65, 145);
  const rightElbowAnchor = setAdaptiveSegment('rightUpperArm', rightShoulderAnchor, rightShoulder, rightElbow, 70, 155);
  const rightHandAnchor = setAdaptiveSegment('rightLowerArm', rightElbowAnchor, rightElbow, rightWrist, 65, 145);
  const leftKneeAnchor = setAdaptiveSegment('leftUpperLeg', leftHipAnchor, leftHip, leftKnee, 85, 175);
  const leftFootAnchor = setAdaptiveSegment('leftLowerLeg', leftKneeAnchor, leftKnee, leftAnkle, 80, 165);
  const rightKneeAnchor = setAdaptiveSegment('rightUpperLeg', rightHipAnchor, rightHip, rightKnee, 85, 175);
  const rightFootAnchor = setAdaptiveSegment('rightLowerLeg', rightKneeAnchor, rightKnee, rightAnkle, 80, 165);

  const earLeft = getPoint(landmarks, 'leftEar');
  const earRight = getPoint(landmarks, 'rightEar');
  const userHead = midpoint(earLeft, earRight);
  const neckAnchor = midpoint(leftShoulderAnchor, rightShoulderAnchor);
  const headDirection = angleBetween(shoulderCenter, userHead);
  const headDistance = clamp(boneLength(shoulderCenter, userHead), 62, 120);
  const headCenter = pointAlong(neckAnchor, headDirection, headDistance);
  // 현재 몸통 너비의 80%로 줄인 지방이 얼굴 비율이에요.
  const headWidth = bodyWidth * 0.8;
  const headAngle = visualLineAngle(earLeft, earRight);
  elements.headPart.setAttribute(
    'transform',
    `translate(${headCenter.x * 1000} ${headCenter.y * 562.5 - headWidth * 0.03}) rotate(${headAngle}) scale(${headWidth / 100})`,
  );

  setPoint('leftHand', leftHandAnchor, angleBetween(leftElbow, leftWrist), 1.05);
  setPoint('rightHand', rightHandAnchor, angleBetween(rightElbow, rightWrist), 1.05);
  setPoint('leftFoot', leftFootAnchor, angleBetween(leftKnee, leftAnkle), 1.08);
  setPoint('rightFoot', rightFootAnchor, angleBetween(rightKnee, rightAnkle), 1.08);
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
refreshCameraList();
if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener?.('devicechange', () => refreshCameraList());
}

window.addEventListener('beforeunload', stopCamera);
