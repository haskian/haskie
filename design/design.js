// shared by every page: loaded right after lucide, before the page's own script
const STATUSBAR_HTML = `<div class="statusbar" role="status">
    <i data-lucide="bot" class="icon"></i>
    <span class="statusbar-item"><span class="muted">Sessions</span><span class="statusbar-counts"><b class="running"><i data-lucide="radio" class="icon"></i>2 active</b><b><i data-lucide="search" class="icon"></i>37 searches today</b></span></span>
    <span class="spacer"></span>
    <i data-lucide="activity" class="icon"></i>
    <span class="statusbar-item"><span class="muted">Jobs</span><span class="statusbar-counts"><b class="done"><i data-lucide="check" class="icon"></i>3</b><b class="running"><i data-lucide="settings" class="icon spin"></i>3</b><b class="queued"><i data-lucide="clock" class="icon"></i>5</b></span>
      <span class="hint" role="tooltip"><span class="hint-section"><span class="label label-mono ok">Running · 3</span><span class="hint-rows"><span>Import</span><span class="muted">renders/lamp.pdf</span><span class="code">38 sec</span><span>Import</span><span class="muted">renders/layers.pdf</span><span class="code">11 sec</span><span>Index</span><span class="muted">collection A–E</span><span class="code">4 sec</span></span></span><span class="hint-section"><span class="label label-mono">Queued · 5</span><span class="hint-rows"><span>Import</span><span class="muted">renders/peak.pdf</span><span class="code">—</span><span>Import</span><span class="muted">renders/penta.pdf</span><span class="code">—</span><span>Import</span><span class="muted">renders/spark.pdf</span><span class="code">—</span><span>Index</span><span class="muted">collection K–O</span><span class="code">—</span><span>Index</span><span class="muted">collection P–T</span><span class="code">—</span></span></span><span class="hint-section"><span class="label label-mono">Done · 3</span><span class="hint-rows"><span>Import</span><span class="muted">renders/cube.pdf</span><span class="code">41 sec</span><span>Import</span><span class="muted">renders/gear.pdf</span><span class="code">58 sec</span><span>Index</span><span class="muted">collection U–Z</span><span class="code">6 sec</span></span></span></span>
    </span>
    <span class="muted">·</span>
    <span class="statusbar-item"><span class="muted">Tasks</span><span class="statusbar-counts"><b class="done"><i data-lucide="check" class="icon"></i>19</b><b class="running"><i data-lucide="settings" class="icon spin"></i>3</b><b class="queued"><i data-lucide="clock" class="icon"></i>41</b></span>
      <span class="hint" role="tooltip"><span class="hint-section"><span class="label label-mono ok">Running · 3</span><span class="hint-rows"><span>Embed chunk 10</span><span class="muted">renders/lamp.pdf</span><span class="code">3 sec</span><span>Chunk chunk 07</span><span class="muted">renders/layers.pdf</span><span class="code">1 sec</span><span>Index chunk 49</span><span class="muted">collection A–E</span><span class="code">0.2 sec</span></span></span><span class="hint-section"><span class="label label-mono">Queued · 41</span><span class="hint-rows"><span>Embed chunk 11</span><span class="muted">renders/lamp.pdf</span><span class="code">—</span><span>Embed chunk 12</span><span class="muted">renders/lamp.pdf</span><span class="code">—</span><span>Chunk chunk 08</span><span class="muted">renders/layers.pdf</span><span class="code">—</span><span>Index chunk 50</span><span class="muted">collection A–E</span><span class="code">—</span><span class="muted" style="grid-column: 1 / -1">+37 more</span></span></span><span class="hint-section"><span class="label label-mono">Done · 19</span><span class="hint-rows"><span>Embed chunk 09</span><span class="muted">renders/lamp.pdf</span><span class="code">3 sec</span><span>Chunk chunk 06</span><span class="muted">renders/layers.pdf</span><span class="code">1 sec</span><span>Index chunk 48</span><span class="muted">collection A–E</span><span class="code">0.2 sec</span><span>Embed chunk 08</span><span class="muted">renders/lamp.pdf</span><span class="code">3 sec</span><span class="muted" style="grid-column: 1 / -1">+15 more</span></span></span></span>
    </span>
  </div>`;
document.getElementById('statusbar').outerHTML = STATUSBAR_HTML;
const LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="275 200 486 600" fill="currentColor"><path d="M761.014,407c-51.328,0 -93,41.672 -93,93c0,-51.328 -41.672,-93 -93,-93c51.328,0 93,-41.672 93,-93c0,51.328 41.672,93 93,93Z"/><rect x="575.014" y="650" width="149.972" height="149.972"/><rect x="425.014" y="500" width="149.972" height="149.972"/><path d="M362.987,200l61.999,0l0,600l-149.972,0l-0,-511.001c0.162,0.001 0.324,0.001 0.486,0.001c48.293,0 87.5,-39.207 87.5,-87.5c0,-0.501 -0.004,-1.001 -0.013,-1.5Z"/><path d="M650.014,649.908l0,0.184c-41.343,0 -74.908,33.565 -74.908,74.908l-0.184,0c0,-41.343 -33.565,-74.908 -74.908,-74.908l0,-0.184c41.343,0 74.908,-33.565 74.908,-74.908l0.184,0c0,41.343 33.565,74.908 74.908,74.908Z"/></svg>';
for (const logo of document.querySelectorAll('.logo')) logo.innerHTML = LOGO_SVG + '<span>haskie</span>';
lucide.createIcons();

// picker: option click copies value + description into summary, closes; outside click closes
for (const picker of document.querySelectorAll('.picker')) {
  for (const option of picker.querySelectorAll('[role="option"]')) {
    const pick = () => {
      for (const other of picker.querySelectorAll('[role="option"]')) other.setAttribute('aria-selected', String(other === option));
      picker.querySelector('.picker-value').innerHTML = option.innerHTML;
      picker.removeAttribute('open');
    };
    option.addEventListener('click', pick);
    option.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pick(); } });
  }
}
document.addEventListener('click', (event) => {
  for (const picker of document.querySelectorAll('.picker[open]')) if (!picker.contains(event.target)) picker.removeAttribute('open');
});

// tabs: one selected per tablist, panels found by aria-controls
function selectTab(tab) {
  for (const other of tab.parentElement.querySelectorAll('.tab')) {
    const on = other === tab;
    other.setAttribute('aria-selected', String(on));
    document.getElementById(other.getAttribute('aria-controls')).hidden = !on;
  }
}
for (const tab of document.querySelectorAll('.tab[aria-controls]')) tab.addEventListener('click', () => selectTab(tab));

// flip a hint to the right edge of its parent when it would run past the viewport
for (const hint of document.querySelectorAll('.hint')) {
  hint.parentElement.addEventListener('mouseenter', () => {
    const left = hint.parentElement.getBoundingClientRect().left;
    hint.classList.toggle('flip', left + hint.offsetWidth > document.documentElement.clientWidth - 16);
  });
}

// modal: a click on the backdrop closes the dialog
for (const dialog of document.querySelectorAll('dialog.modal')) dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
