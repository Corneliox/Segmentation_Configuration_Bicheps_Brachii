/**
 * Biceps Brachii Ultrasound Segmentation - Web Trial (Nano Model)
 * Client-Side Interactive Ultrasound Simulation and Fascia Tracking Engine
 * Scale calibration: 512 px height = 89.96 mm, 219 px width = 38.4 mm (0.1757 mm/pixel)
 */

(function () {
  "use strict";

  const CONFIG = {
    height: 512,
    width: 219,
    mmPerPixel: 0.17569,
    totalFrames: 30,
    fps: 15
  };

  // State
  let currentFrame = 0;
  let isPlaying = false;
  let playInterval = null;
  let activeTool = "brush"; // "brush", "eraser"
  let brushSize = 12;
  let isDrawing = false;
  let showNanoAI = true;
  let showGhost = true;

  // Frame masks storage
  const frameMasks = new Array(CONFIG.totalFrames).fill(null);
  const frameCache = new Array(CONFIG.totalFrames).fill(null);

  // DOM Elements
  const canvas = document.getElementById("viewport");
  const ctx = canvas.getContext("2d");
  const frameSlider = document.getElementById("frameSlider");
  const frameLabel = document.getElementById("frameLabel");
  const btnPlay = document.getElementById("btnPlay");
  const btnPrev = document.getElementById("btnPrev");
  const btnNext = document.getElementById("btnNext");
  const btnNanoAI = document.getElementById("btnNanoAI");
  const btnGhost = document.getElementById("btnGhost");
  const btnBrush = document.getElementById("btnBrush");
  const btnEraser = document.getElementById("btnEraser");
  const btnClear = document.getElementById("btnClear");
  const brushSlider = document.getElementById("brushSlider");
  const brushSizeLabel = document.getElementById("brushSizeLabel");

  const statMeanThickness = document.getElementById("statMeanThickness");
  const statMaxThickness = document.getElementById("statMaxThickness");
  const statFasciaSuperficial = document.getElementById("statFasciaSuperficial");
  const statFasciaDeep = document.getElementById("statFasciaDeep");
  const statMaskArea = document.getElementById("statMaskArea");

  // Pre-generate synthetic B-mode ultrasound frames
  function generateUltrasoundSequence() {
    for (let f = 0; f < CONFIG.totalFrames; f++) {
      const offCanvas = document.createElement("canvas");
      offCanvas.width = CONFIG.width;
      offCanvas.height = CONFIG.height;
      const offCtx = offCanvas.getContext("2d");

      // Background ultrasound speckle noise
      const imgData = offCtx.createImageData(CONFIG.width, CONFIG.height);
      const data = imgData.data;

      // Dynamic contraction factor (sinusoidal)
      const phase = (f / CONFIG.totalFrames) * Math.PI * 2;
      const contraction = Math.sin(phase) * 16;

      // Fascia depths
      const supBase = 150 - contraction * 0.4;
      const deepBase = 370 + contraction * 0.6;

      // Pseudo-random deterministic noise
      let seed = f * 1000;
      function random() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      }

      for (let y = 0; y < CONFIG.height; y++) {
        for (let x = 0; x < CONFIG.width; x++) {
          const idx = (y * CONFIG.width + x) * 4;

          // Acoustic attenuation with depth
          const depthAttenuation = Math.max(0.3, 1.0 - (y / CONFIG.height) * 0.45);

          // Base hypoechoic tissue
          let val = (random() * 32 + random() * 24) * depthAttenuation;

          // Superficial fascia (hyperechoic band around y ~ 140-160)
          const distSup = Math.abs(y - (supBase + Math.sin(x * 0.05) * 4));
          if (distSup < 7) {
            val += (180 - distSup * 22) * (0.8 + random() * 0.4);
          }

          // Deep fascia (hyperechoic band around y ~ 360-380)
          const distDeep = Math.abs(y - (deepBase + Math.cos(x * 0.04) * 6));
          if (distDeep < 8) {
            val += (190 - distDeep * 20) * (0.8 + random() * 0.4);
          }

          // Internal muscle fascicle striations
          if (y > supBase + 10 && y < deepBase - 10) {
            const fascicleAngle = Math.sin((x * 0.8 + y * 0.3) * 0.15);
            if (fascicleAngle > 0.6) {
              val += 35 * random();
            }
          }

          // Subcutaneous fat above superficial fascia
          if (y < supBase - 15) {
            val *= 0.6;
          }

          // Bone / deep acoustic reflection below deep fascia
          if (y > deepBase + 15) {
            const distBone = Math.abs(y - (deepBase + 45));
            if (distBone < 5) val += 150;
            else if (y > deepBase + 50) val *= 0.2; // Acoustic shadow
          }

          val = Math.min(255, Math.max(0, val));
          data[idx] = val;
          data[idx + 1] = val;
          data[idx + 2] = val;
          data[idx + 3] = 255;
        }
      }

      offCtx.putImageData(imgData, 0, 0);
      frameCache[f] = offCanvas;

      // Pre-compute Nano AI Mask for each frame
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = CONFIG.width;
      maskCanvas.height = CONFIG.height;
      const mCtx = maskCanvas.getContext("2d");

      mCtx.fillStyle = "rgba(0, 230, 118, 1.0)";
      mCtx.beginPath();
      mCtx.moveTo(0, supBase);

      for (let x = 0; x <= CONFIG.width; x += 10) {
        mCtx.lineTo(x, supBase + Math.sin(x * 0.05) * 4);
      }
      for (let x = CONFIG.width; x >= 0; x -= 10) {
        mCtx.lineTo(x, deepBase + Math.cos(x * 0.04) * 6);
      }
      mCtx.closePath();
      mCtx.fill();

      frameMasks[f] = maskCanvas;
    }
  }

  // Render current frame, ghost, and active mask onto main viewport
  function render() {
    ctx.clearRect(0, 0, CONFIG.width, CONFIG.height);

    // 1. Draw Raw B-Mode Ultrasound
    const bgFrame = frameCache[currentFrame];
    if (bgFrame) {
      ctx.drawImage(bgFrame, 0, 0);
    }

    // 2. Draw Ghost Layer (Frame n-1) in Amber
    if (showGhost && currentFrame > 0) {
      const prevMask = frameMasks[currentFrame - 1];
      if (prevMask) {
        ctx.save();
        ctx.globalAlpha = 0.35;
        ctx.drawImage(prevMask, 0, 0);
        ctx.strokeStyle = "#ff9800";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(0, 0, 0, 0); // flush
        ctx.restore();
      }
    }

    // 3. Draw Active Frame Mask in Neon Green
    if (showNanoAI) {
      const currMask = frameMasks[currentFrame];
      if (currMask) {
        ctx.save();
        ctx.globalAlpha = 0.40;
        ctx.drawImage(currMask, 0, 0);
        ctx.restore();
      }
    }

    // 4. Update HUD and Measurement Metrics
    updateMetrics();
  }

  // Calculate physical thickness and area metrics
  function updateMetrics() {
    const mask = frameMasks[currentFrame];
    if (!mask) return;

    const mCtx = mask.getContext("2d");
    const imgData = mCtx.getImageData(0, 0, CONFIG.width, CONFIG.height);
    const data = imgData.data;

    let totalThicknessPx = 0;
    let maxThicknessPx = 0;
    let activeColumns = 0;
    let totalActivePixels = 0;
    let meanSupY = 0;
    let meanDeepY = 0;

    for (let x = 0; x < CONFIG.width; x++) {
      let topY = -1;
      let bottomY = -1;
      for (let y = 0; y < CONFIG.height; y++) {
        const idx = (y * CONFIG.width + x) * 4;
        if (data[idx + 3] > 64) {
          totalActivePixels++;
          if (topY === -1) topY = y;
          bottomY = y;
        }
      }

      if (topY !== -1 && bottomY !== -1) {
        const thickness = bottomY - topY;
        totalThicknessPx += thickness;
        if (thickness > maxThicknessPx) maxThicknessPx = thickness;
        meanSupY += topY;
        meanDeepY += bottomY;
        activeColumns++;
      }
    }

    if (activeColumns > 0) {
      const avgThickMm = (totalThicknessPx / activeColumns) * CONFIG.mmPerPixel;
      const maxThickMm = maxThicknessPx * CONFIG.mmPerPixel;
      const supMm = (meanSupY / activeColumns) * CONFIG.mmPerPixel;
      const deepMm = (meanDeepY / activeColumns) * CONFIG.mmPerPixel;
      const areaCm2 = (totalActivePixels * CONFIG.mmPerPixel * CONFIG.mmPerPixel) / 100.0;

      statMeanThickness.textContent = avgThickMm.toFixed(2) + " mm";
      statMaxThickness.textContent = maxThickMm.toFixed(2) + " mm";
      statFasciaSuperficial.textContent = supMm.toFixed(1) + " mm";
      statFasciaDeep.textContent = deepMm.toFixed(1) + " mm";
      statMaskArea.textContent = areaCm2.toFixed(2) + " cm²";
    } else {
      statMeanThickness.textContent = "0.00 mm";
      statMaxThickness.textContent = "0.00 mm";
      statFasciaSuperficial.textContent = "--";
      statFasciaDeep.textContent = "--";
      statMaskArea.textContent = "0.00 cm²";
    }

    frameLabel.textContent = "Frame " + (currentFrame + 1) + " / " + CONFIG.totalFrames;
    frameSlider.value = currentFrame;
  }

  // Brush / Eraser Interaction
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function drawAt(coords) {
    let mask = frameMasks[currentFrame];
    if (!mask) {
      mask = document.createElement("canvas");
      mask.width = CONFIG.width;
      mask.height = CONFIG.height;
      frameMasks[currentFrame] = mask;
    }
    const mCtx = mask.getContext("2d");

    mCtx.save();
    if (activeTool === "brush") {
      mCtx.globalCompositeOperation = "source-over";
      mCtx.fillStyle = "rgba(0, 230, 118, 1.0)";
    } else {
      mCtx.globalCompositeOperation = "destination-out";
      mCtx.fillStyle = "rgba(0, 0, 0, 1.0)";
    }

    mCtx.beginPath();
    mCtx.arc(coords.x, coords.y, brushSize, 0, Math.PI * 2);
    mCtx.fill();
    mCtx.restore();

    render();
  }

  canvas.addEventListener("mousedown", function (e) {
    isDrawing = true;
    drawAt(getCanvasCoords(e));
  });

  window.addEventListener("mousemove", function (e) {
    if (isDrawing) {
      drawAt(getCanvasCoords(e));
    }
  });

  window.addEventListener("mouseup", function () {
    isDrawing = false;
  });

  // Controls Event Binding
  frameSlider.addEventListener("input", function (e) {
    currentFrame = parseInt(e.target.value, 10);
    render();
  });

  btnPlay.addEventListener("click", function () {
    isPlaying = !isPlaying;
    if (isPlaying) {
      btnPlay.textContent = "Pause";
      playInterval = setInterval(function () {
        currentFrame = (currentFrame + 1) % CONFIG.totalFrames;
        render();
      }, 1000 / CONFIG.fps);
    } else {
      btnPlay.textContent = "Play Cine";
      clearInterval(playInterval);
    }
  });

  btnPrev.addEventListener("click", function () {
    if (currentFrame > 0) {
      currentFrame--;
      render();
    }
  });

  btnNext.addEventListener("click", function () {
    if (currentFrame < CONFIG.totalFrames - 1) {
      currentFrame++;
      render();
    }
  });

  btnNanoAI.addEventListener("click", function () {
    showNanoAI = !showNanoAI;
    if (showNanoAI) {
      btnNanoAI.classList.add("active-green");
      btnNanoAI.textContent = "Nano AI: ON";
    } else {
      btnNanoAI.classList.remove("active-green");
      btnNanoAI.textContent = "Nano AI: OFF";
    }
    render();
  });

  btnGhost.addEventListener("click", function () {
    showGhost = !showGhost;
    if (showGhost) {
      btnGhost.classList.add("active-amber");
      btnGhost.textContent = "Ghost Layer: ON";
    } else {
      btnGhost.classList.remove("active-amber");
      btnGhost.textContent = "Ghost Layer: OFF";
    }
    render();
  });

  btnBrush.addEventListener("click", function () {
    activeTool = "brush";
    btnBrush.classList.add("active");
    btnEraser.classList.remove("active");
  });

  btnEraser.addEventListener("click", function () {
    activeTool = "eraser";
    btnEraser.classList.add("active");
    btnBrush.classList.remove("active");
  });

  btnClear.addEventListener("click", function () {
    const mask = frameMasks[currentFrame];
    if (mask) {
      const mCtx = mask.getContext("2d");
      mCtx.clearRect(0, 0, CONFIG.width, CONFIG.height);
      render();
    }
  });

  brushSlider.addEventListener("input", function (e) {
    brushSize = parseInt(e.target.value, 10);
    brushSizeLabel.textContent = brushSize + " px";
  });

  // Initialize
  generateUltrasoundSequence();
  render();
})();
