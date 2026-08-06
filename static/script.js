// ---------- QuanTrip AI frontend logic ----------

const messages = document.getElementById("messages");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const starters = document.getElementById("starters");
const newChatBtn = document.getElementById("newChat");

// one session id per browser tab (this is how the backend remembers you)
let sessionId = "sess-" + Math.random().toString(36).slice(2);

function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = "msg " + who;      // "bot" or "user"
    div.textContent = text;
    messages.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function sendMessage(text, intent = "") {
    if (!text.trim()) return;

    addMessage(text, "user");
    input.value = "";
    sendBtn.disabled = true;

    // simple "typing..." bubble while we wait
    const typing = document.createElement("div");
    typing.className = "msg bot";
    typing.textContent = "Typing…";
    messages.appendChild(typing);
    typing.scrollIntoView({ behavior: "smooth", block: "end" });

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message: text, intent }),
        });
        const data = await res.json();
        typing.remove();
        addMessage(data.reply, "bot");
    } catch (err) {
        typing.remove();
        addMessage("⚠️ Could not reach the server. Is it running?", "bot");
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

sendBtn.addEventListener("click", () => sendMessage(input.value));
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage(input.value);
});

// starter buttons -> send a preset message + set the intent
starters.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    sendMessage(btn.dataset.text, btn.dataset.intent);
});

// new chat -> reset session on the server + clear the screen
newChatBtn.addEventListener("click", async () => {
    await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: "" }),
    });
    sessionId = "sess-" + Math.random().toString(36).slice(2);
    messages.innerHTML =
        '<div class="msg bot">Hi! 👋 New chat started. What kind of trip are you dreaming of?</div>';
});
