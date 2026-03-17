// Background service worker - handles extension lifecycle and tab capture

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'capture-screen-context',
    title: 'Capture Screen for AI Analysis',
    contexts: ['page']
  });
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'capture-screen-context') {
    chrome.runtime.sendMessage({ type: 'trigger-capture' });
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'get-state') {
    sendResponse?.({ ok: true });
    return true;
  }
  
  // Tab capture - sidepanel requests a stream ID for tab audio
  if (msg.type === 'start-tab-capture') {
    const targetTabId = msg.tabId;
    chrome.tabCapture.getMediaStreamId({ targetTabId }, (streamId) => {
      if (chrome.runtime.lastError) {
        sendResponse({ error: chrome.runtime.lastError.message });
      } else {
        sendResponse({ streamId });
      }
    });
    return true; // async response
  }
});
