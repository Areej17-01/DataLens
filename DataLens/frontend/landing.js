const startForm = document.getElementById("start-form");
const nameInput = document.getElementById("name-input");
const startStatus = document.getElementById("start-status");

function buildSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (char) => {
    const rand = Math.floor(Math.random() * 16);
    const value = char === "x" ? rand : (rand & 0x3) | 0x8;
    return value.toString(16);
  });
}

startForm.addEventListener("submit", (event) => {
  event.preventDefault();

  const name = nameInput.value.trim();
  if (!name) {
    startStatus.textContent = "Enter your name first so we can create your workspace.";
    return;
  }

  const sessionId = buildSessionId();
  sessionStorage.setItem("datalensUserName", name);
  sessionStorage.setItem("datalensSessionId", sessionId);
  window.location.href = "/frontend/workspace.html";
});
