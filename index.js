/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as bodyPix from '@tensorflow-models/body-pix';
import Stats from 'stats.js';
import {drawFace} from './demo_util';

const stats = new Stats();

const state = {
  video: null,
  stream: null,
  net: null,
  videoConstraints: {},
  // Triggers the TensorFlow model to reload
  changingArchitecture: false,
  changingMultiplier: false,
  changingStride: false,
  changingResolution: false,
  changingQuantBytes: false,
};

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

async function getVideoInputs() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log('enumerateDevices() not supported.');
    return [];
  }

  const devices = await navigator.mediaDevices.enumerateDevices();

  const videoDevices = devices.filter(device => device.kind === 'videoinput');

  return videoDevices;
}

function stopExistingVideoCapture() {
  if (state.video && state.video.srcObject) {
    state.video.srcObject.getTracks().forEach(track => {
      track.stop();
    })
    state.video.srcObject = null;
  }
}

async function getDeviceIdForLabel(cameraLabel) {
  const videoInputs = await getVideoInputs();

  for (let i = 0; i < videoInputs.length; i++) {
    const videoInput = videoInputs[i];
    if (videoInput.label === cameraLabel) {
      return videoInput.deviceId;
    }
  }

  return null;
}

// on mobile, facing mode is the preferred way to select a camera.
// Here we use the camera label to determine if its the environment or
// user facing camera
function getFacingMode(cameraLabel) {
  if (!cameraLabel) {
    return 'user';
  }
  if (cameraLabel.toLowerCase().includes('back')) {
    return 'environment';
  } else {
    return 'user';
  }
}

async function getConstraints(cameraLabel) {
  let deviceId;
  let facingMode;

  if (cameraLabel) {
    deviceId = await getDeviceIdForLabel(cameraLabel);
    // on mobile, use the facing mode based on the camera.
    facingMode = isMobile() ? getFacingMode(cameraLabel) : null;
  };
  return {deviceId, facingMode};
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera(cameraLabel) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const videoElement = document.getElementById('video');

  stopExistingVideoCapture();

  const videoConstraints = await getConstraints(cameraLabel);

  const stream = await navigator.mediaDevices.getUserMedia(
      {'audio': false, 'video': videoConstraints});
  videoElement.srcObject = stream;

  return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      videoElement.width = videoElement.videoWidth;
      videoElement.height = videoElement.videoHeight;
      resolve(videoElement);
    };
  });
}

async function loadVideo(cameraLabel) {
  try {
    state.video = await setupCamera(cameraLabel);
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  state.video.play();
}

const guiState = {
  algorithm: 'person',
  estimate: 'segmentation',
  camera: null,
  flipHorizontal: true,
  input: {
    architecture: 'MobileNetV1',
    outputStride: 16,
    internalResolution: 'medium',
    multiplier: 0.75,
    quantBytes: 2
  },
  multiPersonDecoding: {
    maxDetections: 20,
    scoreThreshold: 0.9,
    nmsRadius: 20,
    numKeypointForMatching: 17,
    refineSteps: 10
  },
  segmentation: {
    segmentationThreshold: 0.7,
    effect: 'mask',
    maskBackground: true,
    opacity: 1.0,
    backgroundBlurAmount: 3,
    maskBlurAmount: 0,
    edgeBlurAmount: 3
  },
};

function canvasDownload() {
  let canvas = document.getElementById("output");
  let a = document.createElement("a");
  a.href = canvas.toDataURL();
  a.download = "gingerbread-man.png";
  a.click();
}

async function estimateSegmentation() {
  let multiPersonSegmentation = null;
  switch (guiState.algorithm) {
    case 'person':
      return await state.net.segmentPerson(state.video, {
        internalResolution: guiState.input.internalResolution,
        segmentationThreshold: guiState.segmentation.segmentationThreshold,
        maxDetections: guiState.multiPersonDecoding.maxDetections,
        scoreThreshold: guiState.multiPersonDecoding.scoreThreshold,
        nmsRadius: guiState.multiPersonDecoding.nmsRadius,
      });
    default:
      break;
  };
  return multiPersonSegmentation;
}

function drawPoses(personOrPersonPartSegmentation, flipHorizontally, ctx) {
  if (Array.isArray(personOrPersonPartSegmentation)) {
    personOrPersonPartSegmentation.forEach(personSegmentation => {
      let pose = personSegmentation.pose;
      if (flipHorizontally) {
        pose = bodyPix.flipPoseHorizontal(pose, personSegmentation.width);
        const eyepoint1 = pose.keypoints[1];
        const eyepoint2 = pose.keypoints[2];
        drawFace(eyepoint1, eyepoint2, ctx);
      }else{
      const eyepoint1 = pose.keypoints[1];
      const eyepoint2 = pose.keypoints[2];
      drawFace(eyepoint2, eyepoint1, ctx);
      }
    });
  } else {
    personOrPersonPartSegmentation.allPoses.forEach(pose => {
      if (flipHorizontally) {
        pose = bodyPix.flipPoseHorizontal(
            pose, personOrPersonPartSegmentation.width);
        const eyepoint1 = pose.keypoints[1];
        const eyepoint2 = pose.keypoints[2];
        drawFace(eyepoint1, eyepoint2, ctx);
      }
      const eyepoint1 = pose.keypoints[1];
      const eyepoint2 = pose.keypoints[2];
      drawFace(eyepoint2, eyepoint1, ctx);
    })
  }
}

async function loadBodyPix() {
  state.net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2
  });
}

/**
 * Feeds an image to BodyPix to estimate segmentation - this is where the
 * magic happens. This function loops with a requestAnimationFrame method.
 */
function segmentBodyInRealTime() {
  const canvas = document.getElementById('output');
  // since images are being fed from a webcam

  async function bodySegmentationFrame() {
    // Begin monitoring code for frames per second
    stats.begin();

    const flipHorizontally = guiState.flipHorizontal;

    switch (guiState.estimate) {
      case 'segmentation':
        const multiPersonSegmentation = await estimateSegmentation();
        switch (guiState.segmentation.effect) {
          case 'mask':
            const ctx = canvas.getContext('2d');
            const foregroundColor = {r: 170, g: 91, b: 0, a: 255};
            const backgroundColor = {r: 0, g: 0, b: 0, a: 0};
            const mask = bodyPix.toMask(
                multiPersonSegmentation, foregroundColor, backgroundColor,
                false);

            bodyPix.drawMask(
                canvas, state.video, mask, guiState.segmentation.opacity,
                guiState.segmentation.maskBlurAmount, flipHorizontally);
            drawPoses(multiPersonSegmentation, flipHorizontally, ctx);
            break;
        }

        break;
      default:
        break;
    }

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(bodySegmentationFrame);
  }

  bodySegmentationFrame();
}

async function opening() {
  document.getElementById('top-container').style.display = 'flex';
  document.getElementById('top').style.display = 'flex';
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  await sleep(4000);
  document.getElementById('load-container').style.display = 'flex';
  document.getElementById('loading').style.display = 'flex';
  document.getElementById('top-container').style.display = 'none';
  document.getElementById('top').style.display = 'none';
}

/**
 * Kicks off the demo.
 */
export async function bindPage() {
  if(isMobile()) {
    await opening();
  }
  // Load the BodyPix model weights with architecture 0.75
  await loadBodyPix();
  
  await loadVideo(guiState.camera);

  segmentBodyInRealTime();

  document.getElementById('main').style.display = 'inline-block';

  const dl = document.getElementById('download');
  dl.style.display = 'inline-block';
  dl.onclick = (e) => {
    canvasDownload();
  }

  if(isMobile()) {
    const sw = document.getElementById('switch');
    sw.style.display = 'inline-block';
    let switchCounter = 0;
    let cameraLabelMobile = null;
    sw.addEventListener("click", async function() {
      state.changingCamera = true;
      document.getElementById('main').style.display = 'none';
      document.getElementById('load-container').style.display = 'flex';
      document.getElementById('loading').style.display = 'flex';
      switchCounter++;
      if(switchCounter%2==0){
        cameraLabelMobile = null;
      }else{
        cameraLabelMobile = "back";
      }
      
      await loadBodyPix();
 
      await loadVideo(cameraLabelMobile);
      

      if (getFacingMode(cameraLabelMobile) == 'environment') {
        guiState.flipHorizontal = false;
      }else{
        guiState.flipHorizontal = true;
      }
      segmentBodyInRealTime();
      document.getElementById('main').style.display = 'inline-block';
      document.getElementById('load-container').style.display = 'none';
      document.getElementById('loading').style.display = 'none';
      state.changingCamera = false;
    });
  }

  document.getElementById('load-container').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
}


navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
