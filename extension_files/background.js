let popupWindowId = null; // To store the ID of the popup window

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "analyzeImage",
        title: "Analyze Image with AI",
        contexts: ["image"]
    });
});

chrome.contextMenus.onClicked.addListener((info) => {
    if (info.menuItemId === "analyzeImage") {
        chrome.windows.getCurrent({populate: false}, function(currentWindow) {
            const width = 360;
            const height = 500;
            const left = Math.round(currentWindow.left + (currentWindow.width - width) / 2);
            const top = Math.round(currentWindow.top + (currentWindow.height - height) / 2);
            const analyzerUrl = `analyzer.html?image=${encodeURIComponent(info.srcUrl)}`;

            chrome.windows.create({
                url: analyzerUrl,
                type: "popup",
                width: width,
                height: height,
                left: left,
                top: top,
                focused: true
            }, function(createdWindow) {
                popupWindowId = createdWindow.id; // Store the ID of the newly created window
            });
        });
    }
});

chrome.windows.onFocusChanged.addListener(function(windowId) {
    if (popupWindowId !== null && windowId !== popupWindowId && windowId !== chrome.windows.WINDOW_ID_NONE) {
        chrome.windows.get(popupWindowId, function(window) {
            if (chrome.runtime.lastError) {
                // Handle the case where the window does not exist
                console.log("Error: ", chrome.runtime.lastError.message);
                popupWindowId = null; // Reset the window ID if there's an error
            } else if (window) {
                chrome.windows.remove(popupWindowId, function() {
                    if (chrome.runtime.lastError) {
                        // Handle potential error when trying to remove the window
                        console.log("Error removing window: ", chrome.runtime.lastError.message);
                    }
                    popupWindowId = null; // Reset the window ID after closing
                });
            }
        });
    }
});
