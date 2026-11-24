document.addEventListener('DOMContentLoaded', () => {
  const menu = document.getElementById('menuToggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('mobileOverlay');
  const closeMenu = () => { sidebar?.classList.remove('open'); overlay?.classList.remove('visible'); menu?.setAttribute('aria-expanded', 'false'); };
  menu?.addEventListener('click', () => {
    const open = sidebar.classList.toggle('open');
    overlay?.classList.toggle('visible', open);
    menu.setAttribute('aria-expanded', String(open));
  });
  overlay?.addEventListener('click', closeMenu);

  const range = document.getElementById('concentration');
  const rangeLabel = document.getElementById('concentrationLabel');
  range?.addEventListener('input', () => { rangeLabel.textContent = `${range.value} / 10`; });

  const form = document.getElementById('sessionForm');
  if (!form) return;
  const plannedInput = document.getElementById('plannedMinutes');
  const display = document.getElementById('timerDisplay');
  const plannedLabel = document.getElementById('plannedLabel');
  const caption = document.getElementById('timerCaption');
  const progress = document.getElementById('timerProgress');
  const status = document.getElementById('timerStatus');
  const statusDot = document.getElementById('timerStatusDot');
  const message = document.getElementById('sessionMessage');
  const start = document.getElementById('startTimer');
  const pause = document.getElementById('pauseTimer');
  const resume = document.getElementById('resumeTimer');
  const stop = document.getElementById('stopTimer');
  let state = 'ready';
  let interval = null;
  let elapsed = 0;
  let lastTick = 0;
  let goalSeconds = 25 * 60;

  const format = secs => `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(Math.floor(secs % 60)).padStart(2, '0')}`;
  const updateDisplay = () => {
    const remaining = Math.max(0, goalSeconds - elapsed);
    display.textContent = format(remaining);
    progress.style.width = `${Math.min(100, elapsed / goalSeconds * 100)}%`;
    if (remaining === 0 && state === 'running') {
      state = 'finished'; clearInterval(interval); interval = null;
      status.textContent = 'Goal reached — save when ready';
      statusDot.className = 'timer-status-dot'; caption.textContent = 'goal complete';
      pause.disabled = true; resume.hidden = true; stop.disabled = false;
      message.textContent = 'Nice work. Complete your reflection, then save this session.';
    }
  };
  const startTicker = () => {
    lastTick = Date.now();
    interval = setInterval(() => {
      const now = Date.now();
      elapsed += (now - lastTick) / 1000;
      lastTick = now;
      updateDisplay();
    }, 250);
  };
  const updateGoalLabels = () => {
    const mins = Math.max(1, Math.min(1440, Number(plannedInput.value) || 25));
    goalSeconds = mins * 60;
    plannedLabel.textContent = format(goalSeconds);
    if (state === 'ready') { display.textContent = format(goalSeconds); }
  };
  plannedInput.addEventListener('input', updateGoalLabels);
  plannedInput.addEventListener('change', updateGoalLabels);
  document.querySelectorAll('[data-minutes]').forEach(button => button.addEventListener('click', () => {
    if (state !== 'ready') return;
    plannedInput.value = button.dataset.minutes; updateGoalLabels();
  }));

  start.addEventListener('click', () => {
    if (!form.reportValidity()) return;
    updateGoalLabels(); elapsed = 0; state = 'running';
    status.textContent = 'You’re in a focus session'; statusDot.className = 'timer-status-dot running';
    caption.textContent = 'minutes remaining';
    start.disabled = true; plannedInput.disabled = true;
    document.querySelectorAll('[data-minutes]').forEach(button => button.disabled = true);
    pause.disabled = false; stop.disabled = false; message.textContent = '';
    startTicker();
  });
  pause.addEventListener('click', () => {
    if (state !== 'running') return;
    clearInterval(interval); interval = null; state = 'paused';
    status.textContent = 'Paused — take a breath'; statusDot.className = 'timer-status-dot paused';
    pause.hidden = true; resume.hidden = false; caption.textContent = 'session paused';
  });
  resume.addEventListener('click', () => {
    if (state !== 'paused') return;
    state = 'running'; status.textContent = 'You’re in a focus session'; statusDot.className = 'timer-status-dot running';
    pause.hidden = false; resume.hidden = true; caption.textContent = 'minutes remaining'; startTicker();
  });
  stop.addEventListener('click', async () => {
    if (!['running', 'paused', 'finished'].includes(state)) return;
    if (state === 'running') { clearInterval(interval); interval = null; }
    if (elapsed < 1) elapsed = 1;
    state = 'saving'; stop.disabled = true; pause.disabled = true; resume.hidden = true;
    status.textContent = 'Saving your reflection…'; statusDot.className = 'timer-status-dot';
    const payload = {
      subject: document.getElementById('subject').value.trim(),
      goal: document.getElementById('goal').value.trim(),
      planned_minutes: Number(plannedInput.value),
      study_minutes: Math.min(1440, elapsed / 60),
      distraction_minutes: Number(document.getElementById('distractionMinutes').value || 0),
      distraction_count: Number(document.getElementById('distractionCount').value || 0),
      break_minutes: Number(document.getElementById('breakMinutes').value || 0),
      concentration: Number(range.value)
    };
    try {
      const response = await fetch('/api/sessions', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not save this session.');
      state = 'saved'; status.textContent = 'Session saved'; statusDot.className = 'timer-status-dot';
      caption.textContent = 'reflection saved';
      message.className = 'form-message success';
      message.innerHTML = `Saved · <strong>${escapeHtml(result.category)}</strong> model estimate (${Math.round(result.confidence * 100)}% class confidence). <a class="text-link" href="/analysis">View analysis →</a>`;
      stop.textContent = 'Saved';
      window.setTimeout(() => { window.location.href = '/dashboard'; }, 1800);
    } catch (error) {
      state = elapsed >= goalSeconds ? 'finished' : 'paused';
      status.textContent = 'Could not save — check your reflection';
      message.className = 'form-message'; message.textContent = error.message;
      stop.disabled = false;
      if (state === 'paused') { resume.hidden = false; pause.hidden = true; }
    }
  });
});
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
