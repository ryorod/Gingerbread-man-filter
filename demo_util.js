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
import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs-core';

const COLOR = 'aqua';
const BOUNDING_BOX_COLOR = 'red';
const LINE_WIDTH = 2;

export const TRY_RESNET_BUTTON_NAME = 'tryResNetButton';
export const TRY_RESNET_BUTTON_TEXT = '[New] Try ResNet50';
const TRY_RESNET_BUTTON_TEXT_CSS = 'width:100%;text-decoration:underline;';
const TRY_RESNET_BUTTON_BACKGROUND_CSS = 'background:#e61d5f;';

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

function setDatGuiPropertyCss(propertyText, liCssString, spanCssString = '') {
  var spans = document.getElementsByClassName('property-name');
  for (var i = 0; i < spans.length; i++) {
    var text = spans[i].textContent || spans[i].innerText;
    if (text == propertyText) {
      spans[i].parentNode.parentNode.style = liCssString;
      if (spanCssString !== '') {
        spans[i].style = spanCssString;
      }
    }
  }
}

export function updateTryResNetButtonDatGuiCss() {
  setDatGuiPropertyCss(
      TRY_RESNET_BUTTON_TEXT, TRY_RESNET_BUTTON_BACKGROUND_CSS,
      TRY_RESNET_BUTTON_TEXT_CSS);
}

/**
 * Toggles between the loading UI and the main canvas UI.
 */
export function toggleLoadingUI(
    showLoadingUI, loadingDivId = 'loading', mainDivId = 'main') {
  if (showLoadingUI) {
    document.getElementById(loadingDivId).style.display = 'block';
    document.getElementById(mainDivId).style.display = 'none';
  } else {
    document.getElementById(loadingDivId).style.display = 'none';
    document.getElementById(mainDivId).style.display = 'block';
  }
}

export function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

//口の描画
export function drawCurve(ctx, y1, y2, x1, x2, r, color) {
  let radian = Math.atan2(y2 - y1, Math.abs(x2 - x1));
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.beginPath();
  //ctx.moveTo((x1 + x2) / 2, (y1 + y2) / 2);
  //ctx.arc((x1 + x2) / 2, (y1 + y2) / 2, r, Math.PI / 6, Math.PI * 5 / 6);
  ctx.arc((x1 + x2) / 2, (y1 + y2) / 2, r, radian + Math.PI / 6, radian + Math.PI * 5 / 6);
  //ctx.closePath();
  ctx.stroke();
}

//リボンの扇の描画
export function drawRightPartOfRibbon(ctx, y1, y2, x1, x2, r, ribY, color) {
  let radian = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo((x1 + x2) / 2, ribY);
  ctx.arc((x1 + x2) / 2, ribY, r, radian - Math.PI / 6, radian + Math.PI / 6);
  //ctx.arc((x1 + x2) / 2, ribY, r, radian + Math.PI * 5 / 6, radian + Math.PI * 7 / 6);
  ctx.closePath();
  ctx.fill();
}

export function drawLeftPartOfRibbon(ctx, y1, y2, x1, x2, r, ribY, color) {
  let radian = Math.atan2(y2 - y1, x2 - x1);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo((x1 + x2) / 2, ribY);
  //ctx.arc((x1 + x2) / 2, ribY, r, radian - Math.PI / 6, radian + Math.PI / 6);
  ctx.arc((x1 + x2) / 2, ribY, r, radian + Math.PI * 5 / 6, radian + Math.PI * 7 / 6);
  ctx.closePath();
  ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
export function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = LINE_WIDTH;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
  const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  function toTuple({y, x}) {
    return [y, x];
  }

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
        toTuple(keypoints[0].position), toTuple(keypoints[1].position), COLOR,
        scale, ctx);
  });
}

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, COLOR);
  }
}

//目を描画する
export function drawEyes(keypoints, ctx, scale = 1) {
  const eyepoint1 = keypoints[1];
  const eyepoint2 = keypoints[2];
  
  var {y, x} = eyepoint1.position;
  const y1 = y;
  const x1 = x;
  var {y, x} = eyepoint2.position;
  const y2 = y;
  const x2 = x;
  if((x2-x1)>0){
    drawPoint(ctx, y1 * scale, x1 * scale, Math.abs(x2 - x1) / 8, 'white');
    drawPoint(ctx, y2 * scale, x2 * scale, Math.abs(x2 - x1) / 8, 'white');
  }
}

//口を描画する
export function drawMouth(keypoints, ctx, scale = 1) {
  const eyepoint1 = keypoints[1];
  const eyepoint2 = keypoints[2];
  
  var {y, x} = eyepoint1.position;
  const y1 = y;
  const x1 = x;
  var {y, x} = eyepoint2.position;
  const y2 = y;
  const x2 = x;
  if((x2-x1)>0){
    drawCurve(ctx, y1 * scale, y2 * scale, x1 * scale, x2 * scale, Math.abs(x2 - x1), 'white');
  }
}

//リボンを描画する
export function drawRibbon(keypoints, ctx, scale = 1) {
  const eyepoint1 = keypoints[1];
  const eyepoint2 = keypoints[2];
  
  var {y, x} = eyepoint1.position;
  const y1 = y;
  const x1 = x;
  var {y, x} = eyepoint2.position;
  const y2 = y;
  const x2 = x;
  const r = Math.abs(x2 - x1);
  
  if((x2-x1)>0){
    drawPoint(ctx, (((y1 + y2) / 2) + r * 2) * scale, ((x1 + x2) / 2) * scale, Math.abs(x2 - x1) / 4, 'red');
    drawRightPartOfRibbon(ctx, y1 * scale, y2 * scale, x1 * scale, x2 * scale, Math.abs(x2 - x1) * scale, (((y1 + y2) / 2) + r * 2) * scale, 'red');
    drawLeftPartOfRibbon(ctx, y1 * scale, y2 * scale, x1 * scale, x2 * scale, Math.abs(x2 - x1) * scale, (((y1 + y2) / 2) + r * 2) * scale, 'red');
  }
}

export function drawFace(keypoints, ctx, scale = 1) {
  drawEyes(keypoints, ctx, scale);
  drawMouth(keypoints, ctx, scale);
  drawRibbon(keypoints, ctx, scale);
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
export function drawBoundingBox(keypoints, ctx) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);

  ctx.strokeStyle = boundingBoxColor;
  ctx.stroke();
}

/**
 * Converts an array of pixel data into an ImageData object
 */
export async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

/**
 * Draw heatmap values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's heatmap outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawHeatMapValues(heatMapValues, outputStride, canvas) {
  const ctx = canvas.getContext('2d');
  const radius = 5;
  const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

  drawPoints(ctx, scaledValues, radius, COLOR);
}

/**
 * Used by the drawHeatMapValues method to draw heatmap points on to
 * the canvas
 */
function drawPoints(ctx, points, radius, color) {
  const data = points.buffer().values;

  for (let i = 0; i < data.length; i += 2) {
    const pointY = data[i];
    const pointX = data[i + 1];

    if (pointX !== 0 && pointY !== 0) {
      ctx.beginPath();
      ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

/**
 * Draw offset vector values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's offset vector outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawOffsetVectors(
    heatMapValues, offsets, outputStride, scale = 1, ctx) {
  const offsetPoints =
      posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);

  const heatmapData = heatMapValues.buffer().values;
  const offsetPointsData = offsetPoints.buffer().values;

  for (let i = 0; i < heatmapData.length; i += 2) {
    const heatmapY = heatmapData[i] * outputStride;
    const heatmapX = heatmapData[i + 1] * outputStride;
    const offsetPointY = offsetPointsData[i];
    const offsetPointX = offsetPointsData[i + 1];

    drawSegment(
        [heatmapY, heatmapX], [offsetPointY, offsetPointX], COLOR, scale, ctx);
  }
}
