const video = document.querySelector('#camera-video');
const cameraButton = document.querySelector('#camera-button');
const cameraButtonLabel = document.querySelector('#camera-button-label');
const cameraPreview = document.querySelector('#camera-preview');
const cameraStatus = document.querySelector('.camera-status');
const cameraStatusText = document.querySelector('#camera-status-text');
const barkButton = document.querySelector('#bark-button');
const volumeSlider = document.querySelector('#volume');
const volumeValue = document.querySelector('#volume-value');
const dogStage = document.querySelector('#dog-stage');
const stageCaption = document.querySelector('#stage-caption-text');
const moodText = document.querySelector('#mood-text');
const speechBubble = document.querySelector('#speech-bubble');

let faceLandmarker;
let cameraStream;
let audioContext;
let isCameraRunning = false;
let isProcessingFrame = false;
let mouthWasOpen = false;
let lastBarkAt = 0;
let barkTimer;
let activeBarks = 0;
let volume = Number(volumeSlider.value) / 100;

const visionVersion = '0.10.21';
const wasmFiles = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${visionVersion}/wasm`;
const faceModel = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

function setStatus(message, state = '') {
  cameraStatusText.textContent = message;
  cameraStatus.classList.toggle('is-ready', state === 'ready');
  cameraStatus.classList.toggle('is-error', state === 'error');
}

function animateBark() {
  clearTimeout(barkTimer);
  dogStage.classList.add('is-barking');
  speechBubble.textContent = activeBarks % 3 === 1 ? '窖！' : activeBarks % 3 === 2 ? '叫！' : '嚼！';
  stageCaption.textContent = ['好大声！', '咆哮！', '嚼嚼嚼！'][activeBarks % 3];
  moodText.textContent = '很有精神！';
  barkTimer = window.setTimeout(() => {
    dogStage.classList.remove('is-barking');
    stageCaption.textContent = '大狗大狗嚼嚼嚼';
    moodText.textContent = isCameraRunning ? '叮咚基' : '大狗嚼';
  }, 550);
}

async function getAudioContext() {
  if (!audioContext) {
    audioContext = new window.AudioContext();
  }
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }
  return audioContext;
}

async function playBark() {
  activeBarks += 1;
  animateBark();

  try {
    const context = await getAudioContext();
    const now = context.currentTime;
    const master = context.createGain();
    master.gain.value = volume;
    master.connect(context.destination);

    const barkBursts = [0, 0.19];
    barkBursts.forEach((delay, index) => {
      const start = now + delay;
      const burst = context.createGain();
      burst.gain.setValueAtTime(0.0001, start);
      burst.gain.exponentialRampToValueAtTime(index === 0 ? 0.55 : 0.34, start + 0.018);
      burst.gain.exponentialRampToValueAtTime(0.0001, start + 0.15);
      burst.connect(master);

      const voice = context.createOscillator();
      voice.type = 'sawtooth';
      voice.frequency.setValueAtTime(180, start);
      voice.frequency.exponentialRampToValueAtTime(92, start + 0.14);
      voice.connect(burst);
      voice.start(start);
      voice.stop(start + 0.16);

      const noiseBuffer = context.createBuffer(1, Math.ceil(context.sampleRate * 0.18), context.sampleRate);
      const channel = noiseBuffer.getChannelData(0);
      for (let sample = 0; sample < channel.length; sample += 1) {
        channel[sample] = (Math.random() * 2 - 1) * (1 - sample / channel.length);
      }
      const noise = context.createBufferSource();
      const filter = context.createBiquadFilter();
      const noiseGain = context.createGain();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(850, start);
      filter.frequency.exponentialRampToValueAtTime(430, start + 0.14);
      filter.Q.value = 0.8;
      noiseGain.gain.setValueAtTime(0.0001, start);
      noiseGain.gain.exponentialRampToValueAtTime(index === 0 ? 0.42 : 0.25, start + 0.018);
      noiseGain.gain.exponentialRampToValueAtTime(0.0001, start + 0.14);
      noise.buffer = noiseBuffer;
      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(master);
      noise.start(start);
      noise.stop(start + 0.16);
    });
  } catch (error) {
    setStatus('声音启动失败，请检查浏览器设置。', 'error');
  }
}

async function loadFaceLandmarker() {
  if (faceLandmarker) return faceLandmarker;
  setStatus('正在准备嘴型识别…');
  const vision = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${visionVersion}/+esm`);
  const files = await vision.FilesetResolver.forVisionTasks(wasmFiles);
  faceLandmarker = await vision.FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: faceModel, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    minFaceDetectionConfidence: 0.55,
    minFacePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55
  });
  return faceLandmarker;
}

function pointDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function detectMouth() {
  if (!isCameraRunning || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || isProcessingFrame) {
    window.requestAnimationFrame(detectMouth);
    return;
  }

  isProcessingFrame = true;
  try {
    const result = faceLandmarker.detectForVideo(video, performance.now());
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks) {
      setStatus('未识别到人脸，请不要端出来真狗');
      mouthWasOpen = false;
    } else {
      const mouthGap = pointDistance(landmarks[13], landmarks[14]);
      const mouthWidth = pointDistance(landmarks[78], landmarks[308]);
      const openingRatio = mouthGap / mouthWidth;
      const isMouthOpen = mouthWasOpen ? openingRatio > 0.13 : openingRatio > 0.2;

      setStatus(isMouthOpen ? '检测到张嘴！' : '看着镜头，试着张大嘴巴。', 'ready');
      if (isMouthOpen && !mouthWasOpen && performance.now() - lastBarkAt > 850) {
        lastBarkAt = performance.now();
        playBark();
      }
      mouthWasOpen = isMouthOpen;
    }
  } catch (error) {
    console.error('Face detection failed:', error);
    setStatus('识别遇到一点问题，请关闭后重试。', 'error');
    stopCamera(false);
    return;
  } finally {
    isProcessingFrame = false;
  }
  window.requestAnimationFrame(detectMouth);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('请用 localhost 或 HTTPS 打开页面，才能使用摄像头。', 'error');
    return;
  }

  cameraButton.disabled = true;
  cameraButtonLabel.textContent = '正在连接…';
  try {
    await getAudioContext();
    const landmarkerPromise = loadFaceLandmarker();
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = cameraStream;
    await video.play();
    await landmarkerPromise;
    isCameraRunning = true;
    cameraPreview.classList.add('is-live');
    cameraButtonLabel.textContent = '关闭摄像头';
    cameraButton.querySelector('.button-icon').textContent = '◌';
    setStatus('摄像头已开启，试着张嘴吧。', 'ready');
    moodText.textContent = '等你张嘴';
    window.requestAnimationFrame(detectMouth);
  } catch (error) {
    console.error('Could not start camera:', error);
    stopCamera(false);
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      setStatus('没有摄像头权限，可以在浏览器地址栏旁重新允许。', 'error');
    } else if (error.name === 'NotFoundError') {
      setStatus('没有找到可用的摄像头。', 'error');
    } else {
      setStatus('摄像头或嘴型识别启动失败，请检查网络后重试。', 'error');
    }
  } finally {
    cameraButton.disabled = false;
  }
}

function stopCamera(resetMessage = true) {
  isCameraRunning = false;
  mouthWasOpen = false;
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
  }
  video.srcObject = null;
  cameraPreview.classList.remove('is-live');
  cameraButtonLabel.textContent = '开启摄像头';
  cameraButton.querySelector('.button-icon').textContent = '◉';
  moodText.textContent = '等你来玩';
  if (resetMessage) setStatus('摄像头已关闭。');
}

cameraButton.addEventListener('click', () => {
  if (isCameraRunning) stopCamera();
  else startCamera();
});

barkButton.addEventListener('click', playBark);

volumeSlider.addEventListener('input', () => {
  volume = Number(volumeSlider.value) / 100;
  volumeValue.value = `${volumeSlider.value}%`;
  volumeValue.textContent = `${volumeSlider.value}%`;
  volumeSlider.style.setProperty('--volume', `${volumeSlider.value}%`);
});

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && !event.repeat && !['INPUT', 'BUTTON'].includes(document.activeElement.tagName)) {
    event.preventDefault();
    playBark();
  }
});

window.addEventListener('pagehide', () => stopCamera(false));
volumeSlider.style.setProperty('--volume', `${volumeSlider.value}%`);