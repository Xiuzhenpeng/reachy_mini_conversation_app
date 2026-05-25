function setStatus(el, message, kind = "") {
  el.textContent = message || "";
  el.className = kind ? `status ${kind}` : "status";
}

async function getJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

function fieldValue(id) {
  return document.getElementById(id)?.value?.trim() || "";
}

function fillServiceForm(status) {
  document.getElementById("self-openai-base-url").value = "";
  document.getElementById("self-openai-api-key").value = "";
  document.getElementById("self-asr-base-url").value = status.asr_base_url || "";
  document.getElementById("self-asr-model").value = status.asr_model || "";
  document.getElementById("self-llm-base-url").value = status.llm_base_url || "";
  document.getElementById("self-llm-model").value = status.llm_model || "";
  document.getElementById("self-tts-base-url").value = status.tts_base_url || "";
  document.getElementById("self-tts-model").value = status.tts_model || "";
  document.getElementById("self-tts-voice").value = status.tts_voice || "";
  document.getElementById("self-tts-voices").value = Array.isArray(status.tts_voices) ? status.tts_voices.join(",") : "";
  document.getElementById("self-tts-response-format").value = status.tts_response_format || "wav";
}

async function loadPersonalities() {
  const select = document.getElementById("personality-select");
  const data = await getJSON("/personalities");
  select.innerHTML = "";
  for (const choice of data.choices || []) {
    const opt = document.createElement("option");
    opt.value = choice;
    opt.textContent = choice;
    select.appendChild(opt);
  }
  select.value = data.current || data.startup || select.options[0]?.value || "";
}

async function loadVoices() {
  const select = document.getElementById("voice-select");
  const voices = await getJSON("/voices");
  const current = await getJSON("/voices/current");
  select.innerHTML = "";
  for (const voice of voices || []) {
    const opt = document.createElement("option");
    opt.value = voice;
    opt.textContent = voice;
    select.appendChild(opt);
  }
  select.value = current.voice || select.options[0]?.value || "";
}

async function init() {
  const serviceStatus = document.getElementById("status");
  const personalityStatus = document.getElementById("personality-status");
  const form = document.getElementById("service-form");

  try {
    const status = await getJSON("/status");
    fillServiceForm(status);
    setStatus(serviceStatus, `${status.label || "Self-hosted backend"} ready.`, "ok");
  } catch (e) {
    setStatus(serviceStatus, `Could not load service status: ${e.message}`, "error");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(serviceStatus, "Saving...");
    const body = {
      self_openai_base_url: fieldValue("self-openai-base-url"),
      self_openai_api_key: fieldValue("self-openai-api-key"),
      self_asr_base_url: fieldValue("self-asr-base-url"),
      self_asr_model: fieldValue("self-asr-model"),
      self_llm_base_url: fieldValue("self-llm-base-url"),
      self_llm_model: fieldValue("self-llm-model"),
      self_tts_base_url: fieldValue("self-tts-base-url"),
      self_tts_model: fieldValue("self-tts-model"),
      self_tts_voice: fieldValue("self-tts-voice"),
      self_tts_voices: fieldValue("self-tts-voices"),
      self_tts_response_format: fieldValue("self-tts-response-format"),
    };
    try {
      const data = await postJSON("/service_config", body);
      fillServiceForm(data);
      await loadVoices();
      setStatus(serviceStatus, "Saved. Restart the conversation app if a running turn is already using old values.", "ok");
    } catch (e) {
      setStatus(serviceStatus, `Save failed: ${e.message}`, "error");
    }
  });

  try {
    await loadPersonalities();
    await loadVoices();
  } catch (e) {
    setStatus(personalityStatus, `Could not load profiles: ${e.message}`, "error");
  }

  document.getElementById("apply-personality").addEventListener("click", async () => {
    const name = document.getElementById("personality-select").value;
    setStatus(personalityStatus, "Applying...");
    try {
      const data = await postJSON("/personalities/apply", { name, persist: true });
      setStatus(personalityStatus, data.status || "Applied.", "ok");
    } catch (e) {
      setStatus(personalityStatus, `Apply failed: ${e.message}`, "error");
    }
  });

  document.getElementById("apply-voice").addEventListener("click", async () => {
    const voice = document.getElementById("voice-select").value;
    setStatus(personalityStatus, "Applying voice...");
    try {
      const data = await postJSON("/voices/apply", { voice });
      setStatus(personalityStatus, data.status || "Voice applied.", "ok");
    } catch (e) {
      setStatus(personalityStatus, `Voice failed: ${e.message}`, "error");
    }
  });
}

window.addEventListener("DOMContentLoaded", init);
