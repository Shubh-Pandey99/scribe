// Content script — handles desktop notifications from background messages.
// Status overlay and on-page UI are intentionally disabled (stealth design).

chrome.runtime.onMessage.addListener((message) => {
  switch (message.type) {
    case 'recording-started':
      showNotification('Scribe', 'Recording started');
      break;
    case 'recording-stopped':
      showNotification('Scribe', 'Recording stopped');
      break;
    case 'ai-response-ready':
      showNotification('Scribe', 'AI response ready');
      break;
  }
});

function showNotification(title, body) {
  if (Notification.permission === 'granted') {
    new Notification(title, {
      body,
      icon: chrome.runtime.getURL('icon.png'),
    });
  }
}