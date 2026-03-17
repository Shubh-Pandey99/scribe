document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) lucide.createIcons();

  // Elements
  const appContainer = document.getElementById('app');
  const recordBtn = document.getElementById('main-record-btn');
  const recordBtnText = document.getElementById('record-btn-text');
  const micRecordBtn = document.getElementById('mic-record-btn');
  const micRecordBtnText = document.getElementById('mic-record-btn-text');
  const toggleTranscript = document.getElementById('toggle-transcript');
  const toggleCapture = document.getElementById('toggle-capture');
  const backToTranscriptBtn = document.getElementById('back-to-transcript-btn');

  const views = {
    recording: document.getElementById('recording-view'),
    captured: document.getElementById('captured-view'),
    result: document.getElementById('result-view'),
    history: document.getElementById('history-view')
  };

  const cardActionBar = document.getElementById('card-action-bar');
  const processBtn = document.getElementById('process-content-btn');
  const processBtnText = processBtn.querySelector('span');

  const transcriptEl = document.getElementById('live-transcript');
  const meterFill = document.getElementById('meter-fill');
  const imageGallery = document.getElementById('image-gallery');
  const addSnapBtn = document.getElementById('add-snap-btn');
  const clearSnapsBtn = document.getElementById('clear-snaps-btn');
  const aiResponseText = document.getElementById('ai-response-text');
  const errorDisplay = document.getElementById('error-display');

  const askInput = document.getElementById('ask-input');
  const sendBtn = document.getElementById('send-btn');
  const dashboardBtn = document.getElementById('dashboard-btn');
  const settingsBtn = document.getElementById('settings-btn');
  const stealthBtn = document.getElementById('stealth-btn');
  const pauseRecordBtn = document.getElementById('pause-record-btn');
  const settingsOverlay = document.getElementById('settings-overlay');
  const settingsCloseBtn = document.getElementById('settings-close-btn');
  const retakeBtn = document.getElementById('retake-btn');

  // ====== STATE ======
  let isRecording = false;
  let isPaused = false;
  let isStealth = false;
  let aggregatedTranscript = '';
  let activeCaptureDataList = [];
  let currentMode = 'recording';
  let mediaStream = null;
  let mediaRecorder = null;
  let audioContext = null;
  let volumeRaf = null;
  let chunkTimer = null;
  let sessionId = crypto.randomUUID();
  let sessionStart = null;

  function logStatus(text) {
    console.log("[Scribe]", text);
  }

  function setMode(mode) {
    currentMode = mode;
    Object.values(views).forEach(v => v && v.classList.add('hidden'));
    if (views[mode]) views[mode].classList.remove('hidden');

    const isVisionResult = (mode === 'result' && activeCaptureDataList.length > 0);
    const isTransResult = (mode === 'result' && activeCaptureDataList.length === 0);

    toggleTranscript.classList.toggle('active', mode === 'recording' || isTransResult);
    toggleCapture.classList.toggle('active', mode === 'captured' || isVisionResult);

    // Show/hide the bottom action bar
    const showBar = mode === 'recording' || mode === 'captured';
    cardActionBar.classList.toggle('hidden', !showBar);

    if (mode === 'recording') processBtnText.textContent = 'Summarize Transcription';
    else if (mode === 'captured') processBtnText.textContent = 'Analyze Snapshot(s)';

    if (window.lucide) lucide.createIcons();
    updateInputContext();
  }

  function showError(msg) {
    if (!msg) { errorDisplay.classList.add('hidden'); return; }
    errorDisplay.textContent = msg;
    errorDisplay.classList.remove('hidden');
    setTimeout(() => errorDisplay.classList.add('hidden'), 8000);
  }

  function updateInputContext() {
    const hasCapture = activeCaptureDataList.length > 0 && (currentMode === 'captured' || (currentMode === 'result' && activeCaptureDataList.length > 0));
    askInput.placeholder = hasCapture ? "Ask about these captures..." : "Ask anything about this meeting...";
    appContainer.classList.toggle('has-attachment', !!hasCapture);
  }

  function blobToDataURL(blob) {
    return new Promise(resolve => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.readAsDataURL(blob);
    });
  }

  function resizeImageBase64(dataUrl, maxDims = 1920) {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDims || height > maxDims) {
          const ratio = Math.min(maxDims / width, maxDims / height);
          width = Math.round(width * ratio);
          height = Math.round(height * ratio);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        // Fill white background just in case there's transparency (JPEG doesn't support alpha)
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);
        
        resolve(canvas.toDataURL('image/jpeg', 0.80)); // compress to 80% Quality
      };
      img.onerror = () => resolve(dataUrl);
      img.src = dataUrl;
    });
  }

  async function getSettings() {
    return new Promise(resolve => {
      chrome.storage.local.get(['vercelUrl', 'apiKey', 'model'], res => {
        resolve({
          url: res.vercelUrl || 'https://scribe-extension.vercel.app',
          apiKey: res.apiKey || '',
          model: res.model || 'gemini-2.5-flash'
        });
      });
    });
  }

  async function fetchApi(path, options = {}) {
    const s = await getSettings();
    const headers = options.headers || {};
    if (s.apiKey) { headers['X-API-Key'] = s.apiKey; }
    return fetch(s.url + path, { ...options, headers });
  }

  async function saveSession() {
    if (!aggregatedTranscript.trim()) return;
    const payload = {
      id: sessionId,
      transcript: aggregatedTranscript,
      started_at: sessionStart,
      ended_at: new Date().toISOString(),
      title: aggregatedTranscript.split(' ').slice(0, 8).join(' ') + '...'
    };
    // Save locally
    const key = 'session_' + sessionId;
    chrome.storage.local.set({ [key]: payload });
    // Save index
    chrome.storage.local.get(['session_index'], r => {
      const idx = r.session_index || [];
      if (!idx.includes(sessionId)) idx.unshift(sessionId);
      if (idx.length > 30) idx.pop(); // keep last 30
      chrome.storage.local.set({ session_index: idx });
    });
    // Try cloud sync
    try {
      fetchApi('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } catch { }
  }

  // ====== CORE RECORDING ======
  async function startMicRecording() {
    try {
      logStatus("Requesting microphone...");
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioTracks = mediaStream.getAudioTracks();
      if (audioTracks.length === 0) {
        showError("No audio track from microphone.");
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }
      logStatus("✅ Mic audio captured!");
      setupRecording(audioTracks, "mic");
    } catch (err) {
      if (err.name === 'NotAllowedError') logStatus("Mic access denied.");
      else { showError("Mic recording failed: " + err.message); logStatus("Error: " + err.message); }
    }
  }

  async function startRecording() {
    try {
      logStatus("Getting active tab...");

      // Step 1: Get the active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) {
        showError("No active tab found.");
        return;
      }
      logStatus("Tab: " + (tab.title || tab.url).substring(0, 40));

      // Step 2: Get a tabCapture stream ID from the background service worker
      logStatus("Requesting tab audio...");
      const captureResult = await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: 'start-tab-capture', tabId: tab.id }, resolve);
      });

      if (captureResult?.error) {
        // Fallback to getDisplayMedia if tabCapture fails
        logStatus("tabCapture failed: " + captureResult.error + ", trying screen share...");
        return startRecordingFallback();
      }

      // Step 3: Use the stream ID with getUserMedia (reliable audio!)
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          mandatory: {
            chromeMediaSource: 'tab',
            chromeMediaSourceId: captureResult.streamId
          }
        }
      });

      const audioTracks = mediaStream.getAudioTracks();
      if (audioTracks.length === 0) {
        showError("No audio track from tab capture.");
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }

      logStatus("✅ Tab audio captured! Tracks: " + audioTracks.length);
      setupRecording(audioTracks, "tab");

    } catch (err) {
      logStatus("tabCapture error: " + err.message + ", trying fallback...");
      return startRecordingFallback();
    }
  }

  // Fallback: use getDisplayMedia if tabCapture doesn't work
  async function startRecordingFallback() {
    try {
      logStatus("Requesting screen share...");
      mediaStream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true,
        preferCurrentTab: false
      });

      const audioTracks = mediaStream.getAudioTracks();
      if (audioTracks.length === 0) {
        showError("No audio track. Check 'Share tab audio' in the picker.");
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }

      await new Promise(r => setTimeout(r, 300));

      if (audioTracks[0].readyState !== 'live') {
        showError("Audio track ended. Try again.");
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }

      logStatus("✅ Display audio captured!");
      setupRecording(audioTracks, "tab");

    } catch (err) {
      if (err.name === 'NotAllowedError') logStatus("Share cancelled.");
      else { showError("Recording failed: " + err.message); logStatus("Error: " + err.message); }
    }
  }

  // Common setup for recording (used by both tabCapture and getDisplayMedia)
  function setupRecording(audioTracks, mode) {
      isRecording = true;
      aggregatedTranscript = '';
      transcriptEl.textContent = '';
      sessionStart = new Date().toISOString();
      sessionId = crypto.randomUUID();
      appContainer.classList.add('recording');

      isPaused = false;
      if (pauseRecordBtn) {
        pauseRecordBtn.classList.remove('hidden');
        pauseRecordBtn.innerHTML = '<i data-lucide="pause"></i>';
        pauseRecordBtn.style.color = '';
        if (window.lucide) lucide.createIcons();
      }

      if (mode === "mic") {
        micRecordBtnText.textContent = "Stop Mic";
        recordBtn.style.display = "none";
      } else {
        recordBtnText.textContent = "Stop Tab";
        micRecordBtn.style.display = "none";
      }

      // Volume meter
      audioContext = new AudioContext();
      const src = audioContext.createMediaStreamSource(new MediaStream(audioTracks));
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      const dataArr = new Uint8Array(analyser.frequencyBinCount);
      const volumeTick = () => {
        analyser.getByteFrequencyData(dataArr);
        const avg = dataArr.reduce((a, b) => a + b, 0) / dataArr.length / 255;
        if (meterFill) meterFill.style.width = Math.min(100, Math.floor(avg * 300)) + '%';
        volumeRaf = requestAnimationFrame(volumeTick);
      };
      volumeRaf = requestAnimationFrame(volumeTick);
      logStatus("Audio stream active!");

      // MediaRecorder - audio only
      const audioOnlyStream = new MediaStream(audioTracks);
      const configs = [
        [audioOnlyStream, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 64000 }],
        [audioOnlyStream, { mimeType: 'audio/webm', audioBitsPerSecond: 64000 }],
        [audioOnlyStream, { audioBitsPerSecond: 64000 }],
        [audioOnlyStream, {}],
      ];

      getSettings().then(settings => {
        logStatus("API: " + new URL(settings.url).hostname);

        // Chunk queue - process one at a time, buffer the rest
        const chunkQueue = [];
        let processing = false;

        async function processQueue() {
          if (processing || chunkQueue.length === 0) return;
          processing = true;
          const { b64, mime } = chunkQueue.shift();
          try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 15000);
            
            // Append context snippet so ai engine properly stitches word boundaries
            const previousText = aggregatedTranscript.slice(-500); 

            const res = await fetchApi('/api/transcribe', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ audioBase64: b64, mimeType: mime || 'audio/webm', sessionId, previousText }),
              signal: controller.signal
            });
            clearTimeout(timeout);
            if (res.ok) {
              const data = await res.json();
              if (data.error) { logStatus("Error: " + data.error); showError(data.error); return; }
              const text = (data.text || '').trim();
              
              // Whisper models aggressively hallucinate these phrases during pure silence
              // Strip punctuation/spaces to catch repeating loops like "Thank you. Thank you."
              const cleanText = text.toLowerCase().replace(/[^a-z]/g, '');
              const isHallucination = /^(thankyou|thanksforwatching|subtitlesby.*|amaraorg.*|pleasesubscribe|you|bye)+$/.test(cleanText);

              if (text && text.length > 1 && !['SILENT','MUSIC','.'].includes(text.toUpperCase()) && !isHallucination) {
                appendTranscript(data.text);
                logStatus("[" + (data.method || "?") + "] ✓");
              } else {
                // Show debug info so we can see WHY transcription failed
                const reason = isHallucination ? "hallucination dropped" : "no speech";
                const dbg = data.debug ? " | " + data.debug.substring(0, 80) : "";
                logStatus((data.method || "none") + ": " + reason + dbg);
              }
            } else {
              const t = await res.text();
              logStatus("API " + res.status + ": " + t.substring(0, 50));
            }
          } catch (err) {
            if (err.name === 'AbortError') logStatus("⏱ chunk timed out");
            else logStatus("Net: " + err.message);
          }
          processing = false;
          if (chunkQueue.length > 0) processQueue();
        }

        // --- MANAGE RECORDING IN CHUNKS ---
        // We use a start/stop loop instead of .start(5000) so that *every* chunk
        // gets a complete WebM header. Groq will reject chunks that are just raw clusters.

        function recordNextChunk() {
          if (!isRecording) return;
          
          let recorder = null;
          for (const [stream, opts] of configs) {
            try {
              recorder = new MediaRecorder(stream, opts);
              break; // use first working config
            } catch { } // ignore
          }

          if (!recorder) {
            showError("No working recorder on this device.");
            stopRecording();
            return;
          }

          mediaRecorder = recorder; // set global so stopRecording() cleans it up
          mediaRecorder.onerror = (ev) => logStatus("Rec err: " + (ev.error?.message || "?"));

          mediaRecorder.ondataavailable = async (e) => {
            if (!e.data || e.data.size < 100 || isPaused) return; // Drop frame if paused or null
            const curVol = meterFill ? parseInt(meterFill.style.width) || 0 : -1;
            logStatus("Chunk: " + (e.data.size / 1024).toFixed(1) + "KB vol:" + curVol + "%");
            const b64 = await blobToDataURL(e.data);
            if (chunkQueue.length >= 15) chunkQueue.shift(); // 75 second buffer max to prevent transcript drop off
            chunkQueue.push({ b64, mime: e.data.type });
            processQueue();
          };

          mediaRecorder.start(); // No timeslice here

          chunkTimer = setTimeout(() => {
            if (isRecording && mediaRecorder && mediaRecorder.state !== 'inactive') {
              mediaRecorder.stop();
              recordNextChunk(); // Start a new file-chunk immediately
            }
          }, 3000); // 3s segments for faster responsiveness
        }

        recordNextChunk();
        logStatus("🎙 Recording started!");

        // Only stop on audio track ending
        audioTracks[0]?.addEventListener('ended', () => {
          logStatus("⚠ Audio track ended");
          if (chunkTimer) clearTimeout(chunkTimer);
          stopRecording();
        });
      });
  }

  function stopRecording() {
    try { 
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop(); 
      }
    } catch { }
    if (chunkTimer) clearTimeout(chunkTimer);
    try { mediaStream?.getTracks().forEach(t => t.stop()); } catch { }
    try { if (volumeRaf) cancelAnimationFrame(volumeRaf); } catch { }
    try { audioContext?.close(); } catch { }
    mediaRecorder = null;
    mediaStream = null;
    audioContext = null;
    volumeRaf = null;

    if (pauseRecordBtn) pauseRecordBtn.classList.add('hidden');

    isRecording = false;
    appContainer.classList.remove('recording');
    recordBtnText.textContent = "Record Tab";
    micRecordBtnText.textContent = "Record Mic";
    recordBtn.style.display = "";
    micRecordBtn.style.display = "";
    if (meterFill) meterFill.style.width = '0%';
    logStatus("Recording stopped.");
    saveSession();
  }

  function appendTranscript(text) {
    const chunk = text + ' ';
    aggregatedTranscript += chunk;
    const textNode = document.createTextNode(chunk);
    transcriptEl.appendChild(textNode);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }

  // ====== HISTORY VIEW ======
  async function showHistory() {
    setModeRaw('history');
    const histList = document.getElementById('hist-list');
    histList.innerHTML = '<div class="hist-loading">Loading... Cloud History</div>';

    try {
      const res = await fetchApi('/api/sessions');
      if (!res.ok) {
        if (res.status === 401) { throw new Error("Unauthorized. Please set your API Key in Settings."); }
        if (res.status === 429) { throw new Error("Rate Limited."); }
        throw new Error("Failed to load");
      }
      const sessions = await res.json();
      
      if (!sessions || sessions.length === 0) {
        histList.innerHTML = '<div class="hist-empty">No sessions yet.<br>Start recording to create your first one.</div>';
        return;
      }

      histList.innerHTML = '';
      
      sessions.forEach(s => {
        const card = document.createElement('div');
        card.className = 'hist-card';
        
        const date = s.started_at ? new Date(s.started_at).toLocaleDateString('en', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Unknown';
        
        card.innerHTML = `
          <div class="hist-header-row">
            <div class="hist-date">${date}</div>
            <button class="hist-delete-btn" title="Delete session"><i data-lucide="trash-2"></i></button>
          </div>
          <div class="hist-title">${s.title || 'Untitled session'}</div>
          <div class="hist-preview">${(s.transcript || '').substring(0, 80)}...</div>
        `;
        
        // Delete button logic (Cloud)
        const deleteBtn = card.querySelector('.hist-delete-btn');
        deleteBtn.onclick = async (e) => {
          e.stopPropagation(); // Prevent opening the session
          try {
            await fetchApi('/api/sessions/' + s.id, { method: 'DELETE' });
            showHistory(); // Refresh view
          } catch(err) {
            showError("Failed to delete session.");
          }
        };

        // Open session logic
        card.onclick = () => {
          sessionId = s.id;
          aggregatedTranscript = s.transcript || '';
          transcriptEl.textContent = aggregatedTranscript;
          setMode('recording');
        };
        
        histList.appendChild(card);
      });
      
      if (window.lucide) lucide.createIcons();

    } catch (err) {
      histList.innerHTML = '<div class="hist-empty" style="color:#f85149">Could not connect to Cloud Database.</div>';
    }
  }

  // Raw mode set without toggling action bar logic for history
  function setModeRaw(mode) {
    currentMode = mode;
    Object.values(views).forEach(v => v && v.classList.add('hidden'));
    if (views[mode]) views[mode].classList.remove('hidden');
    toggleTranscript.classList.remove('active');
    toggleCapture.classList.remove('active');
    cardActionBar.classList.add('hidden');
    if (window.lucide) lucide.createIcons();
  }

  // ====== UI HANDLERS ======
  recordBtn.onclick = () => {
    if (!isRecording) startRecording();
    else stopRecording();
  };
  micRecordBtn.onclick = () => {
    if (!isRecording) startMicRecording();
    else stopRecording();
  };

  function renderGallery() {
    if (!imageGallery) return;
    imageGallery.innerHTML = '';
    activeCaptureDataList.forEach((dataUrl, index) => {
      const wrap = document.createElement('div');
      wrap.className = 'gallery-item';
      
      const img = document.createElement('img');
      img.src = dataUrl;
      
      const delBtn = document.createElement('button');
      delBtn.className = 'gallery-del-btn';
      delBtn.innerHTML = '<i data-lucide="x"></i>';
      delBtn.onclick = (e) => {
        e.stopPropagation();
        activeCaptureDataList.splice(index, 1);
        renderGallery();
        if (activeCaptureDataList.length === 0) setMode('recording');
      };

      wrap.appendChild(img);
      wrap.appendChild(delBtn);
      imageGallery.appendChild(wrap);
    });
    if (window.lucide) lucide.createIcons();
  }

  async function captureScreen() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return;
      chrome.tabs.captureVisibleTab(tab.windowId, { format: 'jpeg', quality: 90 }, async (dataUrl) => {
        if (chrome.runtime.lastError) { showError("Snap failed: " + chrome.runtime.lastError.message); return; }
        // Compress scale on large 4k screens to prevent 413 Payload Too large on backend
        let captured = await resizeImageBase64(dataUrl, 1600); 
        activeCaptureDataList.push(captured);
        renderGallery();
        setMode('captured');
      });
    } catch (e) { showError("Snap error: " + e.message); }
  }

  async function runAIAction(prompt, extra = {}) {
    if (!extra.imageArray || extra.imageArray.length === 0) activeCaptureDataList = [];
    setMode('result');
    aiResponseText.innerHTML = '<span class="thinking-text">Thinking...</span>';
    try {
      const settings = await getSettings();
      const res = await fetchApi('/api/answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'google', model: settings.model || 'gemini-2.5-flash', transcript: prompt, ...extra })
      });
      if (res.status === 401) { showError("Unauthorized. Check API Key in settings."); return; }
      if (res.status === 429) { showError("Rate Limited by Backend."); return; }

      const data = await res.json();
      if (data.answer) {
        if (window.marked) {
          aiResponseText.innerHTML = marked.parse(data.answer);
          if (window.hljs) {
            aiResponseText.querySelectorAll('pre code').forEach((block) => {
              hljs.highlightElement(block);
            });
          }
        } else {
          aiResponseText.innerHTML = data.answer
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
        }
      } else {
        aiResponseText.textContent = data.error || 'No response from AI.';
      }
    } catch (e) {
      showError('API Error: ' + e.message);
      aiResponseText.textContent = 'Error contacting AI.';
    }
  }

  processBtn.onclick = () => {
    if (currentMode === 'recording') {
      const text = transcriptEl.innerText || aggregatedTranscript;
      if (!text.trim()) { showError('No transcript to summarize yet.'); return; }
      runAIAction('Summarize this meeting transcript concisely in bullet points:\n' + text);
    } else if (currentMode === 'captured') {
      runAIAction('Analyze these screen captures and explain what is shown, highlight any key info.', { imageArray: activeCaptureDataList });
    }
  };

  toggleTranscript.onclick = () => { activeCaptureDataList = []; setMode('recording'); };
  toggleCapture.onclick = () => {
    if (activeCaptureDataList.length > 0 && currentMode !== 'captured') setMode('captured');
    else captureScreen();
  };
  backToTranscriptBtn.onclick = () => { activeCaptureDataList = []; setMode('recording'); };

  if (addSnapBtn) addSnapBtn.onclick = captureScreen;
  if (clearSnapsBtn) clearSnapsBtn.onclick = () => {
    activeCaptureDataList = [];
    renderGallery();
    setMode('recording');
  };

  sendBtn.onclick = () => {
    let q = askInput.value.trim();
    if (!q) return;
    askInput.value = '';

    // Automatically inject the live transcript context into the user's question!
    // Slice off the last ~1000 words (or chunk of logic) so Gemini doesn't 500-error crash on infinite transcripts.
    let contextText = transcriptEl.innerText || aggregatedTranscript;
    if (contextText.trim()) {
      if (contextText.length > 8000) {
        contextText = "..." + contextText.slice(-8000);
      }
      q = q + "\n\n--- Meeting / Interview Context ---\n" + contextText;
    }

    const extra = activeCaptureDataList.length > 0 ? { imageArray: activeCaptureDataList } : {};
    runAIAction(q, extra);
  };
  askInput.onkeypress = (e) => { if (e.key === 'Enter') sendBtn.click(); };

  // Stealth mode toggle
  if (stealthBtn) {
    stealthBtn.onclick = () => {
      isStealth = !isStealth;
      appContainer.classList.toggle('stealth-mode', isStealth);
      stealthBtn.innerHTML = `<i data-lucide="${isStealth ? 'eye-off' : 'eye'}"></i>`;
      if (window.lucide) lucide.createIcons();
    };
  }

  // Pause Mic toggle
  if (pauseRecordBtn) {
    pauseRecordBtn.onclick = () => {
      isPaused = !isPaused;
      if (isPaused) {
        pauseRecordBtn.innerHTML = '<i data-lucide="play"></i>';
        pauseRecordBtn.style.color = '#f85149';
        logStatus("Recording paused.");
      } else {
        pauseRecordBtn.innerHTML = '<i data-lucide="pause"></i>';
        pauseRecordBtn.style.color = '';
        logStatus("Recording resumed.");
      }
      if (window.lucide) lucide.createIcons();
    };
  }

  // Native Clipboard support
  document.addEventListener('paste', async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    let imagePasted = false;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            const blob = items[i].getAsFile();
            const dataUrl = await blobToDataURL(blob);
            const resized = await resizeImageBase64(dataUrl, 1600);
            activeCaptureDataList.push(resized);
            imagePasted = true;
        }
    }
    if (imagePasted) {
        renderGallery();
        if (currentMode !== 'captured') setMode('captured');
    }
  });

  // History button — show local sessions panel
  dashboardBtn.onclick = () => showHistory();
  document.getElementById('hist-back-btn')?.addEventListener('click', () => setMode('recording'));

  // Settings modal
  const modelSelect = document.getElementById('model-select');
  const urlInput = document.getElementById('vercel-url-input');
  const apiKeyInput = document.getElementById('api-key-input');
  settingsBtn.onclick = () => { settingsOverlay.classList.remove('hidden'); if (window.lucide) lucide.createIcons(); };
  settingsCloseBtn.onclick = () => settingsOverlay.classList.add('hidden');
  settingsOverlay.onclick = (e) => { if (e.target === settingsOverlay) settingsOverlay.classList.add('hidden'); };
  document.getElementById('save-settings').onclick = () => {
    chrome.storage.local.set({ model: modelSelect.value, vercelUrl: urlInput.value, apiKey: apiKeyInput.value }, () => {
      logStatus('Settings saved.');
      settingsOverlay.classList.add('hidden');
    });
  };
  chrome.storage.local.get(['model', 'vercelUrl', 'apiKey'], (res) => {
    if (res.model) modelSelect.value = res.model;
    if (res.vercelUrl) urlInput.value = res.vercelUrl;
    if (res.apiKey) apiKeyInput.value = res.apiKey;
  });

  if (retakeBtn) retakeBtn.onclick = () => captureScreen();

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'error') { showError(msg.message); logStatus('Error: ' + msg.message); }
    if (msg.type === 'trigger-capture') captureScreen();
  });

  logStatus('System ready.');
});
