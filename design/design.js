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
const LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1230 711" fill="currentColor"><path d="M660,24 L662,24 L673,38 L684,51 L695,65 L704,76 L715,90 L728,106 L739,120 L750,134 L761,148 L772,162 L778,170 L781,171 L883,171 L893,184 L907,202 L920,219 L932,234 L945,251 L959,269 L970,283 L980,296 L993,313 L1000,322 L1018,329 L1081,352 L1138,373 L1201,396 L1205,398 L1205,450 L1194,463 L1186,473 L1172,490 L1161,504 L1150,517 L1139,531 L1128,544 L1120,554 L1108,569 L1097,583 L1092,589 L1076,591 L905,606 L828,613 L814,622 L795,634 L771,649 L752,661 L733,673 L713,686 L593,686 L597,683 L626,666 L641,657 L670,640 L697,624 L726,607 L755,590 L782,574 L808,559 L809,558 L1009,539 L1061,534 L1070,522 L1086,501 L1099,484 L1113,466 L1123,453 L1133,440 L1142,428 L1135,425 L1083,409 L1016,388 L951,368 L946,366 L934,350 L920,331 L904,310 L888,289 L879,277 L869,264 L859,251 L847,235 L843,230 L841,229 L761,225 L748,211 L739,200 L729,189 L720,178 L707,164 L698,153 L686,140 L679,132 L670,122 L667,118 L667,254 L761,348 L756,352 L741,362 L722,374 L689,396 L673,406 L667,410 L676,416 L702,432 L722,444 L729,449 L724,453 L707,462 L684,475 L657,490 L634,503 L609,517 L584,531 L559,545 L535,559 L513,571 L492,583 L465,598 L444,610 L428,619 L419,616 L376,597 L340,581 L299,563 L261,546 L220,528 L206,522 L24,521 L28,517 L47,503 L68,487 L86,474 L107,458 L122,447 L141,433 L157,421 L176,407 L192,395 L211,381 L229,368 L248,354 L254,350 L261,351 L305,365 L309,364 L328,345 L336,338 L367,307 L375,300 L403,272 L411,265 L433,243 L441,236 L464,213 L472,206 L494,184 L502,177 L523,156 L531,149 L553,127 L561,120 L583,98 L591,91 L615,67 L623,60 L643,40 L651,33 Z "/></svg>';
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
