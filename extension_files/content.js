// content.js
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "getScreenDimensions") {
        sendResponse({
            screenWidth: window.screen.width,
            screenHeight: window.screen.height
        });
    }
});
