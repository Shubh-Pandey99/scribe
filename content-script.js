// Content script for on-page status overlay and notifications
let statusOverlay = null;
let isRecording = false;

// Create status overlay
function createStatusOverlay() {
  if (statusOverlay) return;

  statusOverlay = document.createElement('div');
  statusOverlay.id = 'ai-meeting-status-overlay';
  statusOverlay.innerHTML = `
    <div id="ai-status-content">
      <div id="ai-status-icon">🤖</div>
      <div id="ai-status-text">AI Assistant Ready</div>
    </div>
  `;

  // Add styles
  const styles = `
    #ai-meeting-status-overlay {
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 10000;
      background: rgba(18, 18, 18, 0.95);
      color: white;
      border: 1px solid #2A2A2A;
      border-radius: 8px;
      padding: 12px 16px;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
      opacity: 0;
      transform: translateY(-10px);
    }
    
    #ai-meeting-status-overlay.visible {
      opacity: 1;
      transform: translateY(0);
    }
    
    #ai-meeting-status-overlay.recording {
      background: rgba(29, 185, 84, 0.95);
      border-color: #1DB954;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(29, 185, 84, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(29, 185, 84, 0); }
      100% { box-shadow: 0 0 0 0 rgba(29, 185, 84, 0); }
    }
    
    #ai-status-content {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    #ai-status-icon {
      font-size: 16px;
    }
  `;

  const styleSheet = document.createElement('style');
  styleSheet.textContent = styles;
  document.head.appendChild(styleSheet);

  document.body.appendChild(statusOverlay);

  // Show overlay
  setTimeout(() => statusOverlay.classList.add('visible'), 100);
}

// Update status overlay
function updateStatusOverlay(status, recording = false) {
  if (!statusOverlay) createStatusOverlay();

  const statusText = document.getElementById('ai-status-text');
  const statusIcon = document.getElementById('ai-status-icon');

  statusText.textContent = status;

  if (recording) {
    statusOverlay.classList.add('recording');
    statusIcon.textContent = '🎙️';
  } else {
    statusOverlay.classList.remove('recording');
    statusIcon.textContent = '🤖';
  }

  isRecording = recording;
}

// Hide status overlay
function hideStatusOverlay() {
  if (statusOverlay) {
    statusOverlay.classList.remove('visible');
    setTimeout(() => {
      if (statusOverlay) {
        statusOverlay.remove();
        statusOverlay = null;
      }
    }, 300);
  }
}

// Show desktop notification
function showDesktopNotification(title, body, icon = null) {
  if (Notification.permission === 'granted') {
    new Notification(title, {
      body: body,
      icon: icon || chrome.runtime.getURL('icon.png'),
      badge: chrome.runtime.getURL('icon.png')
    });
  }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'recording-started':
      // updateStatusOverlay('Recording Active', true);
      showDesktopNotification('AI Meeting Assistant', 'Recording started');
      break;

    case 'recording-stopped':
      // updateStatusOverlay('AI Assistant Ready', false);
      showDesktopNotification('AI Meeting Assistant', 'Recording stopped');
      break;

    case 'transcript-update':
      // updateStatusOverlay('Transcribing...', true);
      break;

    case 'ai-response-ready':
      showDesktopNotification('AI Meeting Assistant', 'AI response ready');
      break;

    case 'show-status':
      // createStatusOverlay();
      break;

    case 'hide-status':
      // hideStatusOverlay();
      break;
  }
});

// Initialize when content script loads
// createStatusOverlay(); // Disabled as per user request