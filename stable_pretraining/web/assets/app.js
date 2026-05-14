// spt-web: a tiny wandb-like viewer.
// Pure vanilla JS + uPlot. No build step.

(() => {
  const uPlot = window.uPlot;

  // ---- state ------------------------------------------------------------

  const state = {
    runs: new Map(),         // run_id -> sidecar summary
    metrics: new Map(),      // run_id -> {metrics: {name: {step, epoch, y}}}
    media: new Map(),        // run_id -> [event, ...]  (parsed media.jsonl)
    visible: new Set(),      // visible run_ids
    charts: new Map(),       // metric_name -> {panel, plot, configKey}
    mediaPanels: new Map(),  // tag -> {panel, type, step}
    smoothing: 0,
    xAxis: 'step',
    logY: false,
    search: '',
    groupBy: '',             // '' (none) | 'status' | 'hparams.X'
    sortBy: 'created_at',
    sortDesc: true,
    detailRunId: null,       // run currently shown in the modal
    detailFilter: '',
    filters: [],             // [{key, values: [string,...]}], AND across, OR within
    expandedMetrics: new Set(),  // metric tree paths the user has expanded (closed by default)
    openSidebarGroups: new Set(), // sidebar group keys the user has expanded
    theme: 'dark',
    activeTab: 'figures',   // 'figures' | 'out' | 'err'
    metricSearch: '',
    // Cached per-run log discoveries: runId -> {streams, fetchedAt}
    logsIndex: new Map(),
    // Currently selected (runId, stream_id) for each kind, persisted across
    // tab switches so the user doesn't lose their place.
    logSelection: { out: null, err: null },  // null or {runId, streamId}
  };

  const SYNC_KEY = 'sptweb-x';

  // ---- theme -----------------------------------------------------------

  const SUN_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round">'
    + '<circle cx="8" cy="8" r="3"/>'
    + '<line x1="8" y1="1" x2="8" y2="3"/>'
    + '<line x1="8" y1="13" x2="8" y2="15"/>'
    + '<line x1="1" y1="8" x2="3" y2="8"/>'
    + '<line x1="13" y1="8" x2="15" y2="8"/>'
    + '<line x1="3" y1="3" x2="4.5" y2="4.5"/>'
    + '<line x1="11.5" y1="11.5" x2="13" y2="13"/>'
    + '<line x1="3" y1="13" x2="4.5" y2="11.5"/>'
    + '<line x1="11.5" y1="4.5" x2="13" y2="3"/>'
    + '</svg>';
  const MOON_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">'
    + '<path d="M14 9.5A6 6 0 1 1 6.5 2a4.5 4.5 0 0 0 7.5 7.5z"/>'
    + '</svg>';

  function themeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
  }

  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('spt-web-theme', theme); } catch {}
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.innerHTML = theme === 'dark' ? SUN_SVG : MOON_SVG;
    // Force chart rebuild: configKey reset triggers fresh uPlot creation
    // which picks up the new themed colors via getComputedStyle.
    for (const e of state.charts.values()) e.configKey = '';
    colorCache.clear();
    if (state.runs.size > 0) renderRunList();
    if (state.detailRunId) renderDetail();
    scheduleRerender();
  }

  function initTheme() {
    let saved = 'dark';
    try { saved = localStorage.getItem('spt-web-theme') || 'dark'; } catch {}
    applyTheme(saved);
  }

  // ---- color: stable hash → HSL ----------------------------------------

  const colorCache = new Map();
  function runColor(id) {
    let c = colorCache.get(id);
    if (c) return c;
    let h = 2166136261;
    for (let i = 0; i < id.length; i++) {
      h ^= id.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const hue = Math.abs(h) % 360;
    // Brighter on dark, slightly darker/saturated on light for contrast.
    const isLight = state.theme === 'light';
    const sat = isLight ? '70%' : '75%';
    const lum = isLight ? '42%' : '62%';
    c = `hsl(${hue} ${sat} ${lum})`;
    colorCache.set(id, c);
    return c;
  }

  // ---- API --------------------------------------------------------------

  async function fetchJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  async function refreshRuns() {
    const runs = await fetchJSON('/api/runs');
    const seen = new Set();
    for (const r of runs) {
      seen.add(r.run_id);
      state.runs.set(r.run_id, r);
    }
    for (const id of [...state.runs.keys()]) {
      if (!seen.has(id)) {
        state.runs.delete(id);
        state.metrics.delete(id);
        state.visible.delete(id);
        state.logsIndex.delete(id);
      }
    }
    renderRunList();
    updateHeaderStats();
  }

  // Stream-fetch metrics via NDJSON. The server emits one JSON object per
  // line; each object is either a chunk of new points or {done:true}.
  // We append into state.metrics as chunks arrive and re-render every few
  // chunks, so the chart paints within a few hundred ms even for runs with
  // huge metrics.csv files.
  async function fetchMetrics(runId) {
    // Build the new payload in a *separate* dict and only commit it to
    // ``state.metrics`` once the stream completes. If we wipe ``state.metrics``
    // up-front, the chart blanks for the duration of the stream — the user
    // sees the page flash on every SSE update during training.
    // For the first-ever fetch (no existing data) we still commit early so
    // progressive paint works; the live-fill UX is only relevant when we
    // have nothing to show yet.
    const isFirst = !state.metrics.has(runId);
    let target;
    if (isFirst) {
      state.metrics.set(runId, { metrics: {} });
      target = state.metrics.get(runId).metrics;
    } else {
      // Re-stream into a buffer; replace at the end.
      target = {};
    }
    let resp;
    try {
      resp = await fetch(
        `/api/metrics-stream?run_id=${encodeURIComponent(runId)}`,
        { cache: 'no-store' },
      );
    } catch (e) {
      console.warn('metrics-stream fetch failed', runId, e);
      return;
    }
    if (!resp.ok || !resp.body) {
      console.warn('metrics-stream not ok', runId, resp.status);
      return;
    }
    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf  = '';
    let chunksSinceRender = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          let obj;
          try { obj = JSON.parse(line); } catch (e) { continue; }
          if (obj.done) continue;
          if (!obj.metrics) continue;
          for (const [name, m] of Object.entries(obj.metrics)) {
            const t = target[name] || (target[name] = { step: [], epoch: [], y: [] });
            // Direct push loop — push-spread blows the stack for very
            // large chunks (5k+ points × N metrics).
            const ms = m.step  || [];
            const me = m.epoch || [];
            const my = m.y     || [];
            for (let i = 0; i < my.length; i++) {
              t.step.push(ms[i]);
              t.epoch.push(me[i]);
              t.y.push(my[i]);
            }
          }
          chunksSinceRender++;
          if (isFirst && chunksSinceRender >= 1) {
            // First fetch: live-fill the chart as chunks arrive.
            chunksSinceRender = 0;
            scheduleRerender();
          }
        }
      }
    } catch (e) {
      console.warn('metrics-stream read failed', runId, e);
    }
    if (!isFirst) {
      // Commit the buffered payload atomically so the chart doesn't blink.
      state.metrics.set(runId, { metrics: target });
      scheduleRerender();
    }
  }

  async function fetchMedia(runId) {
    try {
      const r = await fetchJSON(`/api/media?run_id=${encodeURIComponent(runId)}`);
      state.media.set(runId, r.events || []);
    } catch (e) {
      // No media for this run is normal — keep an empty array so we don't refetch.
      state.media.set(runId, []);
    }
  }

  // ---- search match -----------------------------------------------------

  function matches(run, q) {
    if (!q) return true;
    if (run.run_id.toLowerCase().includes(q)) return true;
    if ((run.tags || []).some(t => String(t).toLowerCase().includes(q))) return true;
    if ((run.notes || '').toLowerCase().includes(q)) return true;
    for (const [k, v] of Object.entries(run.hparams || {})) {
      if (k.toLowerCase().includes(q)) return true;
      if (String(v).toLowerCase().includes(q)) return true;
    }
    return false;
  }

  // ---- filters ---------------------------------------------------------

  function filterableKeys() {
    const keys = new Set(['status']);
    for (const r of state.runs.values()) {
      if ((r.tags || []).length) keys.add('tags');
      for (const k of Object.keys(r.hparams || {})) keys.add(`hparams.${k}`);
      for (const k of Object.keys(r.summary || {})) keys.add(`summary.${k}`);
    }
    return [...keys].sort();
  }

  function distinctValues(key) {
    const vals = new Set();
    for (const r of state.runs.values()) {
      if (key === 'tags') {
        for (const t of r.tags || []) vals.add(String(t));
        continue;
      }
      const v = valueAt(r, key);
      if (v == null || v === '') continue;
      if (Array.isArray(v)) for (const x of v) vals.add(String(x));
      else vals.add(String(v));
    }
    return [...vals].sort((a, b) => {
      const an = Number(a), bn = Number(b);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
      return a.localeCompare(b);
    });
  }

  function runMatchesFilter(run, filter) {
    const set = new Set(filter.values);
    if (filter.key === 'tags') {
      for (const t of run.tags || []) if (set.has(String(t))) return true;
      return false;
    }
    const v = valueAt(run, filter.key);
    if (v == null) return false;
    if (Array.isArray(v)) return v.some(x => set.has(String(x)));
    return set.has(String(v));
  }

  function passesFilters(run) {
    for (const f of state.filters) if (!runMatchesFilter(run, f)) return false;
    return true;
  }

  function effectivelyVisible() {
    return [...state.visible].filter(id => {
      const r = state.runs.get(id);
      return r && passesFilters(r);
    });
  }

  // ---- value extraction / sort / group ---------------------------------

  function valueAt(run, key) {
    if (!key) return null;
    if (key.indexOf('.') < 0) return run[key];
    const dot = key.indexOf('.');
    const ns = key.slice(0, dot);
    const k = key.slice(dot + 1);
    const obj = run[ns];
    return obj ? obj[k] : null;
  }

  function compareValues(a, b) {
    if (a == null && b == null) return 0;
    if (a == null) return 1;   // nulls last regardless of direction
    if (b == null) return -1;
    const an = typeof a === 'number' ? a : Number(a);
    const bn = typeof b === 'number' ? b : Number(b);
    if (!Number.isNaN(an) && !Number.isNaN(bn) && a !== '' && b !== '') {
      return an < bn ? -1 : an > bn ? 1 : 0;
    }
    return String(a).localeCompare(String(b));
  }

  function sortRuns(runs) {
    const key = state.sortBy;
    const sign = state.sortDesc ? -1 : 1;
    return runs.slice().sort((a, b) => {
      const v = compareValues(valueAt(a, key), valueAt(b, key));
      // Nulls-last is direction-independent, so don't apply sign when one is null.
      if (valueAt(a, key) == null || valueAt(b, key) == null) return v;
      return v * sign;
    });
  }

  function partitionRuns(runs) {
    if (!state.groupBy) return { groups: [], ungrouped: runs };
    const groups = new Map();
    const ungrouped = [];
    for (const r of runs) {
      const v = valueAt(r, state.groupBy);
      if (v == null || v === '') {
        ungrouped.push(r);
        continue;
      }
      const key = String(v);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    return {
      groups: [...groups.entries()]
        .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
        .map(([key, rs]) => ({ key, runs: rs })),
      ungrouped,
    };
  }

  // ---- dropdown options ------------------------------------------------

  function refreshKeyOptions() {
    const groupKeys = new Set(['status']);
    const sortKeys = new Set(['created_at', 'run_id', 'status']);
    for (const r of state.runs.values()) {
      for (const k of Object.keys(r.hparams || {})) {
        groupKeys.add(`hparams.${k}`);
        sortKeys.add(`hparams.${k}`);
      }
      for (const k of Object.keys(r.summary || {})) {
        sortKeys.add(`summary.${k}`);
      }
    }
    fillSelect('group-by', ['', ...[...groupKeys].sort()], state.groupBy, v => v || 'none');
    fillSelect('sort-by', [...sortKeys].sort(), state.sortBy);
  }

  function fillSelect(id, options, current, label) {
    const sel = document.getElementById(id);
    const sig = options.join('');
    if (sel.dataset.opts !== sig) {
      sel.dataset.opts = sig;
      sel.replaceChildren(...options.map(v => {
        const o = document.createElement('option');
        o.value = v;
        o.textContent = label ? label(v) : v;
        return o;
      }));
    }
    if (sel.value !== current) sel.value = current;
  }

  // ---- icons -----------------------------------------------------------

  const INFO_SVG = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none">' +
    '<circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.2"/>' +
    '<line x1="8" y1="6.8" x2="8" y2="11.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>' +
    '<circle cx="8" cy="4.4" r="0.85" fill="currentColor"/>' +
    '</svg>';

  // ---- filter rendering ------------------------------------------------

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function renderFilters() {
    const list = document.getElementById('filters-list');
    const frag = document.createDocumentFragment();
    state.filters.forEach((f, idx) => {
      const chip = document.createElement('div');
      chip.className = 'filter-chip';
      chip.title = 'click to edit';

      const k = document.createElement('span');
      k.className = 'filter-chip-key';
      k.textContent = f.key + ':';
      chip.appendChild(k);

      const v = document.createElement('span');
      v.className = 'filter-chip-vals';
      const shown = f.values.slice(0, 3).join(', ');
      v.textContent = shown + (f.values.length > 3 ? ` (+${f.values.length - 3})` : '');
      v.title = f.values.join(', ');
      chip.appendChild(v);

      const rm = document.createElement('button');
      rm.type = 'button';
      rm.className = 'filter-chip-remove';
      rm.textContent = '×';
      rm.title = 'remove filter';
      rm.addEventListener('click', e => {
        e.stopPropagation();
        state.filters.splice(idx, 1);
        renderFilters();
        renderRunList();
        scheduleRerender();
      });
      chip.appendChild(rm);

      chip.addEventListener('click', () => openFilterDraft(idx));
      frag.appendChild(chip);
    });
    list.replaceChildren(frag);
  }

  function openFilterDraft(editIdx = null) {
    const draft = document.getElementById('filter-draft');
    draft.replaceChildren();
    draft.hidden = false;

    const editing = editIdx != null ? state.filters[editIdx] : null;
    const keys = filterableKeys();

    const keySel = document.createElement('select');
    keySel.appendChild(new Option('select key…', ''));
    for (const k of keys) keySel.appendChild(new Option(k, k));
    if (editing) keySel.value = editing.key;
    draft.appendChild(keySel);

    const vals = document.createElement('div');
    vals.className = 'filter-values-picker';
    draft.appendChild(vals);

    const buttons = document.createElement('div');
    buttons.className = 'filter-draft-buttons';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = 'cancel';
    const apply = document.createElement('button');
    apply.type = 'button';
    apply.className = 'primary';
    apply.textContent = editing ? 'update' : 'add';
    apply.disabled = true;
    buttons.appendChild(cancel);
    buttons.appendChild(apply);
    draft.appendChild(buttons);

    function refreshValues() {
      const key = keySel.value;
      vals.replaceChildren();
      apply.disabled = true;
      if (!key) return;
      const distinct = distinctValues(key);
      const preset = new Set(editing && editing.key === key ? editing.values : []);
      for (const dv of distinct) {
        const lbl = document.createElement('label');
        lbl.className = 'filter-value-cb';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = dv;
        cb.checked = preset.has(dv);
        cb.addEventListener('change', () => {
          apply.disabled = vals.querySelectorAll('input:checked').length === 0;
        });
        lbl.appendChild(cb);
        const txt = document.createElement('span');
        txt.textContent = dv;
        lbl.appendChild(txt);
        vals.appendChild(lbl);
      }
      apply.disabled = preset.size === 0;
    }
    keySel.addEventListener('change', refreshValues);
    refreshValues();

    cancel.addEventListener('click', () => {
      draft.hidden = true;
      draft.replaceChildren();
    });
    apply.addEventListener('click', () => {
      const key = keySel.value;
      if (!key) return;
      const checked = [...vals.querySelectorAll('input:checked')].map(cb => cb.value);
      if (!checked.length) return;
      const next = { key, values: checked };
      if (editIdx != null) state.filters[editIdx] = next;
      else state.filters.push(next);
      draft.hidden = true;
      draft.replaceChildren();
      renderFilters();
      renderRunList();
      scheduleRerender();
    });
  }

  // ---- render: run list ------------------------------------------------

  function makeRunRow(r) {
    const row = document.createElement('div');
    const visible = state.visible.has(r.run_id);
    const loading = _loading.has(r.run_id);
    row.className =
      'run-item' + (visible ? ' visible' : '') + (loading ? ' loading' : '');
    row.dataset.runId = r.run_id;

    const dot = document.createElement('div');
    dot.className = 'run-dot';
    dot.style.background = runColor(r.run_id);
    row.appendChild(dot);

    const name = document.createElement('div');
    name.className = 'run-name';
    name.title = `${r.run_id}\n${r.run_dir || ''}`;
    name.textContent = r.run_id;
    row.appendChild(name);

    if (r.status) {
      const st = document.createElement('div');
      st.className = `run-status ${r.status}`;
      st.textContent = r.status;
      row.appendChild(st);
    }

    const info = document.createElement('button');
    info.type = 'button';
    info.className = 'run-info';
    info.title = 'show config';
    info.innerHTML = INFO_SVG;
    info.addEventListener('click', e => {
      e.stopPropagation();
      openDetail(r.run_id);
    });
    row.appendChild(info);

    row.addEventListener('click', () => toggleRun(r.run_id));
    return row;
  }

  function renderRunList() {
    refreshKeyOptions();
    renderFilters();
    const el = document.getElementById('run-list');
    const q = state.search.toLowerCase();
    const filtered = [...state.runs.values()].filter(r => passesFilters(r) && matches(r, q));
    const sorted = sortRuns(filtered);
    const { groups, ungrouped } = partitionRuns(sorted);

    document.getElementById('run-count').textContent =
      `${state.runs.size} run${state.runs.size === 1 ? '' : 's'}` +
      (filtered.length !== state.runs.size ? ` (${filtered.length} shown)` : '');

    // Skip rebuilding the sidebar DOM when the *structure* (group keys,
    // run order, run count) hasn't changed. Background SSE updates fire
    // every few seconds during training; rebuilding 2 000+ rows on every
    // tick is what causes the visible page-flash. Per-row state changes
    // that affect appearance (visibility, loading-pulse) are repainted by
    // touching only the affected row's classList below.
    const layoutKey = JSON.stringify({
      groupKeys: groups.map(g => [g.key, g.runs.map(r => r.run_id)]),
      ungrouped: ungrouped.map(r => r.run_id),
      open: [...state.openSidebarGroups].sort(),
    });
    if (state._lastListKey === layoutKey) {
      // Same layout — just sync the per-row visible/loading classes.
      for (const r of filtered) {
        const row = el.querySelector(`.run-item[data-run-id="${CSS.escape(r.run_id)}"]`);
        if (!row) continue;
        row.classList.toggle('visible', state.visible.has(r.run_id));
        row.classList.toggle('loading', _loading.has(r.run_id));
      }
      return;
    }
    state._lastListKey = layoutKey;

    const frag = document.createDocumentFragment();
    // Real groups first, collapsed by default unless the user opened them.
    for (const { key, runs } of groups) {
      const det = document.createElement('details');
      det.className = 'run-group';
      det.open = state.openSidebarGroups.has(key);
      const sm = document.createElement('summary');
      sm.append(`${key} `);
      const c = document.createElement('span');
      c.className = 'group-count';
      c.textContent = `(${runs.length})`;
      sm.appendChild(c);
      det.appendChild(sm);
      for (const r of runs) det.appendChild(makeRunRow(r));
      det.addEventListener('toggle', () => {
        if (det.open) state.openSidebarGroups.add(key);
        else state.openSidebarGroups.delete(key);
      });
      frag.appendChild(det);
    }
    // Ungrouped runs (or all runs when groupBy is empty) flat at the bottom.
    for (const r of ungrouped) frag.appendChild(makeRunRow(r));
    el.replaceChildren(frag);
  }

  // Run rows currently fetching their metrics/media. Used to render a
  // 'loading' visual state synchronously on click — otherwise users get
  // no feedback during the seconds-long fetch of a multi-MiB metrics CSV.
  const _loading = new Set();

  async function toggleRun(id) {
    if (state.visible.has(id)) {
      state.visible.delete(id);
      renderRunList();
      scheduleRerender();
      return;
    }
    state.visible.add(id);
    const tasks = [];
    if (!state.metrics.has(id)) tasks.push(fetchMetrics(id));
    if (!state.media.has(id)) {
      const r = state.runs.get(id);
      if (r && r.has_media) tasks.push(fetchMedia(id));
      else state.media.set(id, []);
    }
    if (tasks.length) {
      _loading.add(id);
      // Re-render synchronously so the row gets the 'loading' class
      // before the await suspends the function.
      renderRunList();
      try {
        await Promise.all(tasks);
      } finally {
        _loading.delete(id);
      }
    }
    renderRunList();
    scheduleRerender();
    if (state.activeTab === 'out' || state.activeTab === 'err') {
      // Selection changed → log-stream options change.
      fetchLogStreams(id).then(() => renderLogTab(state.activeTab));
    }
  }

  async function setAllVisible(visible) {
    if (!visible) {
      state.visible.clear();
    } else {
      const q = state.search.toLowerCase();
      const ids = [...state.runs.values()]
        .filter(r => passesFilters(r) && matches(r, q))
        .map(r => r.run_id);
      for (const id of ids) state.visible.add(id);
      const needMetrics = ids.filter(id => !state.metrics.has(id));
      const needMedia = ids.filter(id => {
        if (state.media.has(id)) return false;
        const r = state.runs.get(id);
        return r && r.has_media;
      });
      await Promise.all([
        ...needMetrics.map(fetchMetrics),
        ...needMedia.map(fetchMedia),
      ]);
    }
    renderRunList();
    scheduleRerender();
    if (state.activeTab === 'out' || state.activeTab === 'err') {
      refreshLogStreamsForVisibleRuns().then(() => renderLogTab(state.activeTab));
    }
  }

  // ---- render: charts ---------------------------------------------------

  let renderPending = false;
  function scheduleRerender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      renderCharts();
    });
  }

  function visibleMetricNames() {
    const names = new Set();
    for (const id of effectivelyVisible()) {
      const m = state.metrics.get(id);
      if (!m) continue;
      for (const k of Object.keys(m.metrics)) names.add(k);
    }
    const all = [...names].sort();
    const q = (state.metricSearch || '').trim().toLowerCase();
    if (!q) return all;
    const filtered = all.filter(n => n.toLowerCase().includes(q));
    // Update the "M of N matches" status if the input exists.
    const status = document.getElementById('metric-search-status');
    if (status) {
      status.textContent = filtered.length === all.length
        ? `${all.length}`
        : `${filtered.length} / ${all.length} match`;
    }
    return filtered;
  }

  function visibleMediaTags() {
    // tag → 'image' | 'video' (last seen wins; in practice they're stable)
    const out = new Map();
    for (const id of effectivelyVisible()) {
      const events = state.media.get(id);
      if (!events) continue;
      for (const e of events) {
        if (!e || !e.tag) continue;
        out.set(e.tag, e.type || 'image');
      }
    }
    return out;
  }

  function buildMetricTree(names) {
    // Tree node: {children: Map<segment, node>, leaves: [{leafName, fullName}]}
    const root = { children: new Map(), leaves: [] };
    for (const name of names) {
      const parts = name.split('/').filter(p => p !== '');
      if (parts.length === 0) continue;
      let node = root;
      for (let i = 0; i < parts.length - 1; i++) {
        const seg = parts[i];
        if (!node.children.has(seg)) {
          node.children.set(seg, { children: new Map(), leaves: [] });
        }
        node = node.children.get(seg);
      }
      node.leaves.push({ leafName: parts[parts.length - 1], fullName: name });
    }
    return root;
  }

  function countLeaves(node) {
    let n = node.leaves.length;
    for (const c of node.children.values()) n += countLeaves(c);
    return n;
  }

  function attachMetricTree(parent, node, pathPrefix, mediaTags) {
    const items = [
      ...[...node.children.entries()].map(([k, n]) => ({ kind: 'group', key: k, node: n })),
      ...node.leaves.map(l => ({ kind: 'leaf', key: l.leafName, fullName: l.fullName })),
    ].sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));

    for (const item of items) {
      if (item.kind === 'group') {
        const path = pathPrefix ? `${pathPrefix}/${item.key}` : item.key;
        const det = document.createElement('details');
        det.className = 'metric-group';
        // Closed by default; track explicit user expansion in the set.
        det.open = state.expandedMetrics.has(path);

        const sm = document.createElement('summary');
        const lab = document.createElement('span');
        lab.className = 'metric-group-name';
        lab.textContent = item.key;
        const cnt = document.createElement('span');
        cnt.className = 'metric-group-count';
        cnt.textContent = `(${countLeaves(item.node)})`;
        sm.appendChild(lab);
        sm.appendChild(cnt);
        det.appendChild(sm);

        const body = document.createElement('div');
        body.className = 'metric-group-body';
        det.appendChild(body);
        det.addEventListener('toggle', () => {
          if (det.open) state.expandedMetrics.add(path);
          else state.expandedMetrics.delete(path);
        });

        parent.appendChild(det);
        attachMetricTree(body, item.node, path, mediaTags);
      } else {
        const isMedia = mediaTags.has(item.fullName);
        if (isMedia) {
          let entry = state.mediaPanels.get(item.fullName);
          const type = mediaTags.get(item.fullName);
          if (!entry) {
            const panel = makeMediaPanel(item.fullName, item.key, type);
            entry = { panel, type, step: null, lastUrls: new Map() };
            state.mediaPanels.set(item.fullName, entry);
          } else {
            // Refresh the panel title in case it moved within the tree.
            const titleSpan = entry.panel.querySelector('.chart-title .name');
            if (titleSpan) titleSpan.textContent = item.key;
            entry.type = type;
          }
          parent.appendChild(entry.panel);
        } else {
          let entry = state.charts.get(item.fullName);
          if (!entry) {
            const panel = makePanel(item.fullName, item.key);
            entry = { panel, plot: null, configKey: '' };
            state.charts.set(item.fullName, entry);
          } else {
            const titleSpan = entry.panel.querySelector('.chart-title .name');
            if (titleSpan) titleSpan.textContent = item.key;
          }
          parent.appendChild(entry.panel);
        }
      }
    }
  }

  function renderCharts() {
    const root = document.getElementById('charts');
    const metrics = visibleMetricNames();
    const mediaTags = visibleMediaTags();
    const allTags = [...new Set([...metrics, ...mediaTags.keys()])];

    if (allTags.length === 0) {
      for (const { plot } of state.charts.values()) plot?.destroy();
      state.charts.clear();
      for (const { panel } of state.mediaPanels.values()) panel.remove();
      state.mediaPanels.clear();
      state._lastTreeKey = null; // force rebuild next time we leave overview
      renderOverview(root);
      return;
    }

    // Tear down chart panels for metrics that vanished.
    const metricSet = new Set(metrics);
    for (const [name, entry] of [...state.charts.entries()]) {
      if (!metricSet.has(name) || mediaTags.has(name)) {
        entry.plot?.destroy();
        entry.panel.remove();
        state.charts.delete(name);
      }
    }
    // Tear down media panels for tags that vanished.
    for (const [tag, entry] of [...state.mediaPanels.entries()]) {
      if (!mediaTags.has(tag)) {
        entry.panel.remove();
        state.mediaPanels.delete(tag);
      }
    }

    // Skip the full tree rebuild when nothing structurally changed.
    // Streaming-metrics chunks fire scheduleRerender many times per
    // second; re-parenting a playing ``<video>`` (or just shifting page
    // layout) confuses the browser's scroll anchoring and makes the page
    // jump to the top while the user is interacting with media.
    // Only rebuild when the *set* of tags or grouping changed.
    const treeKey = JSON.stringify({
      tags: allTags.slice().sort(),
      groups: state.groupBy ?? null,
    });
    if (state._lastTreeKey !== treeKey) {
      const tree = buildMetricTree(allTags);
      const container = document.createElement('div');
      container.className = 'metric-tree';
      attachMetricTree(container, tree, '', mediaTags);
      root.replaceChildren(container);
      state._lastTreeKey = treeKey;

      // After re-parenting, sizes may have changed. Resize each plot.
      for (const entry of state.charts.values()) {
        if (entry.plot) {
          const body = entry.panel.querySelector('.chart-body');
          const w = body.clientWidth;
          if (w) entry.plot.setSize({ width: w, height: 240 });
        }
      }
    }

    for (const name of metrics) updateChart(name);
    for (const tag of mediaTags.keys()) updateMediaPanel(tag);
  }

  function makePanel(fullName, displayName) {
    const panel = document.createElement('div');
    panel.className = 'chart-panel';
    const title = document.createElement('div');
    title.className = 'chart-title';
    const span = document.createElement('span');
    span.className = 'name';
    span.textContent = displayName || fullName;
    span.title = fullName;
    title.appendChild(span);
    panel.appendChild(title);
    const body = document.createElement('div');
    body.className = 'chart-body';
    panel.appendChild(body);
    return panel;
  }

  // ---- media panels -----------------------------------------------------

  function makeMediaPanel(fullTag, displayName, type) {
    const panel = document.createElement('div');
    panel.className = 'chart-panel media-panel';
    panel.dataset.tag = fullTag;
    panel.dataset.type = type;

    const title = document.createElement('div');
    title.className = 'chart-title';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = displayName || fullTag;
    name.title = fullTag;
    title.appendChild(name);
    const stepInfo = document.createElement('span');
    stepInfo.className = 'media-step-info';
    title.appendChild(stepInfo);
    panel.appendChild(title);

    const sliderWrap = document.createElement('div');
    sliderWrap.className = 'media-slider-wrap';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'media-slider';
    slider.min = '0';
    slider.max = '0';
    slider.value = '0';
    slider.step = '1';
    sliderWrap.appendChild(slider);
    panel.appendChild(sliderWrap);

    const body = document.createElement('div');
    body.className = 'media-body';
    panel.appendChild(body);
    return panel;
  }

  function updateMediaPanel(tag) {
    const entry = state.mediaPanels.get(tag);
    if (!entry) return;

    // Per visible run, the events for this tag, sorted by step ascending.
    const visibleIds = effectivelyVisible();
    const perRun = [];
    for (const id of visibleIds) {
      const all = state.media.get(id) || [];
      const evs = all.filter(e => e.tag === tag);
      if (!evs.length) continue;
      evs.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
      perRun.push({ runId: id, events: evs });
    }

    const body = entry.panel.querySelector('.media-body');
    const slider = entry.panel.querySelector('.media-slider');
    const stepInfo = entry.panel.querySelector('.media-step-info');

    if (perRun.length === 0) {
      body.replaceChildren();
      stepInfo.textContent = '';
      slider.disabled = true;
      return;
    }

    // Union of step values across runs.
    const stepSet = new Set();
    for (const { events } of perRun) for (const e of events) stepSet.add(e.step ?? 0);
    const steps = [...stepSet].sort((a, b) => a - b);

    // Default to the latest seen step on first render or if previous step
    // isn't in the union anymore.
    if (entry.step == null || !stepSet.has(entry.step)) {
      entry.step = steps[steps.length - 1];
    }
    slider.disabled = steps.length <= 1;
    slider.min = '0';
    slider.max = String(steps.length - 1);
    slider.value = String(steps.indexOf(entry.step));
    stepInfo.textContent = `step ${entry.step}`;

    slider.oninput = () => {
      entry.step = steps[parseInt(slider.value, 10)];
      stepInfo.textContent = `step ${entry.step}`;
      renderMediaBody(entry, perRun);
    };

    renderMediaBody(entry, perRun);
  }

  function renderMediaBody(entry, perRun) {
    // Update row DOM **in place** wherever possible. Recreating the
    // ``<img>`` / ``<video>`` element on every slider tick caused the
    // browser to drop the loaded media, collapse the panel to zero
    // height while the new src loads, and reflow the page — which the
    // user sees as a jump back to the top. Reusing the existing element
    // and just swapping ``.src`` keeps dimensions stable across ticks.
    const body = entry.panel.querySelector('.media-body');
    if (!entry.rowEls) entry.rowEls = new Map(); // runId -> {row, head, stepEl, mediaEl, type, captionEl}

    const seen = new Set();
    for (const { runId, events } of perRun) {
      seen.add(runId);

      // Latest event with step <= entry.step (events are step-ascending).
      let chosen = null;
      for (const e of events) {
        if ((e.step ?? 0) <= entry.step) chosen = e;
        else break;
      }

      let r = entry.rowEls.get(runId);
      if (!r) {
        const row = document.createElement('div');
        row.className = 'media-row';
        const head = document.createElement('div');
        head.className = 'media-row-head';
        const dot = document.createElement('span');
        dot.className = 'run-dot';
        dot.style.background = runColor(runId);
        head.appendChild(dot);
        const lab = document.createElement('span');
        lab.className = 'media-row-label';
        lab.textContent = runId;
        lab.title = runId;
        head.appendChild(lab);
        const stepEl = document.createElement('span');
        stepEl.className = 'media-row-step';
        head.appendChild(stepEl);
        row.appendChild(head);
        body.appendChild(row);
        r = { row, head, stepEl, mediaEl: null, type: null, captionEl: null };
        entry.rowEls.set(runId, r);
      }

      if (chosen) {
        r.stepEl.textContent = `@${chosen.step}`;
        const url = `/api/media-file?run_id=${encodeURIComponent(runId)}&path=${encodeURIComponent(chosen.path)}`;
        // Reuse the existing media element when its type matches; just
        // swap src. We compare against a custom ``dataset.spurl`` rather
        // than ``mediaEl.src`` — the browser canonicalises ``.src`` to an
        // absolute URL so naive equality always fails, and re-setting the
        // src on a playing ``<video>`` restarts it from frame 0 (the "bleep"
        // the user noticed when audio cuts off).
        if (r.mediaEl && r.type === chosen.type) {
          if (r.mediaEl.dataset.spurl !== url) {
            r.mediaEl.src = url;
            r.mediaEl.dataset.spurl = url;
            if (chosen.type === 'image') {
              r.mediaEl.alt = `${runId} @ step ${chosen.step}`;
            }
          }
        } else {
          if (r.mediaEl) r.mediaEl.remove();
          let mediaEl;
          if (chosen.type === 'video') {
            mediaEl = document.createElement('video');
            mediaEl.controls = true;
            mediaEl.preload = 'metadata';
            mediaEl.muted = true;
            mediaEl.playsInline = true;
          } else {
            mediaEl = document.createElement('img');
            mediaEl.loading = 'lazy';
            mediaEl.decoding = 'async';
            mediaEl.alt = `${runId} @ step ${chosen.step}`;
            // Click → open in fullscreen lightbox. Videos already have
            // a native fullscreen button via ``controls``; this gives
            // images parity.
            mediaEl.addEventListener('click', () => {
              const cap = chosen.caption
                ? `${runId} @ step ${chosen.step} — ${chosen.caption}`
                : `${runId} @ step ${chosen.step}`;
              openLightbox(mediaEl.src, cap);
            });
          }
          mediaEl.className = 'media-content';
          mediaEl.src = url;
          mediaEl.dataset.spurl = url;
          r.row.appendChild(mediaEl);
          r.mediaEl = mediaEl;
          r.type = chosen.type;
        }
        // Caption (create / update / remove)
        if (chosen.caption) {
          if (!r.captionEl) {
            r.captionEl = document.createElement('div');
            r.captionEl.className = 'media-caption';
            r.row.appendChild(r.captionEl);
          }
          r.captionEl.textContent = chosen.caption;
        } else if (r.captionEl) {
          r.captionEl.remove();
          r.captionEl = null;
        }
        entry.lastUrls.set(runId, url);
      } else {
        // No data at this step — clear any media + show placeholder text.
        r.stepEl.textContent = '';
        if (r.mediaEl) {
          r.mediaEl.remove();
          r.mediaEl = null;
          r.type = null;
        }
        if (!r.captionEl) {
          r.captionEl = document.createElement('div');
          r.captionEl.className = 'media-empty';
          r.row.appendChild(r.captionEl);
        }
        r.captionEl.textContent = 'no data at this step';
      }
    }

    // Drop rows for runs that vanished from selection.
    for (const [runId, r] of entry.rowEls) {
      if (!seen.has(runId)) {
        r.row.remove();
        entry.rowEls.delete(runId);
      }
    }
  }

  function emaSmooth(ys, alpha) {
    if (alpha <= 0) return ys;
    const out = new Array(ys.length);
    let prev = null;
    for (let i = 0; i < ys.length; i++) {
      const v = ys[i];
      if (v == null || !isFinite(v)) {
        out[i] = v;
        continue;
      }
      prev = prev == null ? v : alpha * prev + (1 - alpha) * v;
      out[i] = prev;
    }
    return out;
  }

  function buildChartData(name, runIds) {
    // Per-run x/y arrays for the chosen x-axis.
    const series = [];
    const xUnion = new Set();
    for (const id of runIds) {
      const m = state.metrics.get(id);
      if (!m || !m.metrics[name]) continue;
      const col = m.metrics[name];
      const xs = col[state.xAxis];
      const ys = col.y;
      // Some runs may lack the chosen axis (e.g. epoch missing). Fall back to step.
      const fallback = (state.xAxis === 'epoch' && !xs?.some(v => v != null))
        ? col.step : xs;
      const xArr = fallback || col.step;
      const filtered = { id, xs: [], ys: [] };
      for (let i = 0; i < ys.length; i++) {
        const x = xArr ? xArr[i] : i;
        if (x == null || !isFinite(x)) continue;
        filtered.xs.push(x);
        filtered.ys.push(ys[i]);
        xUnion.add(x);
      }
      if (filtered.ys.length) series.push(filtered);
    }

    const xs = [...xUnion].sort((a, b) => a - b);
    const xIdx = new Map();
    for (let i = 0; i < xs.length; i++) xIdx.set(xs[i], i);

    const seriesCfg = [{}];
    const data = [xs];
    const gKey = state.groupBy
      ? state.groupBy.replace(/^hparams\./, '').replace(/^summary\./, '')
      : null;
    for (const s of series) {
      const arr = new Array(xs.length).fill(null);
      for (let i = 0; i < s.xs.length; i++) arr[xIdx.get(s.xs[i])] = s.ys[i];
      data.push(emaSmooth(arr, state.smoothing));
      let label = s.id;
      if (gKey) {
        const run = state.runs.get(s.id);
        const gv = run ? valueAt(run, state.groupBy) : null;
        const gShown = gv == null || gv === '' ? '(none)' : String(gv);
        label = `${gKey}=${gShown} · ${s.id}`;
      }
      seriesCfg.push({
        label,
        runId: s.id, // used by the in-chart tooltip as the row label
        stroke: runColor(s.id),
        width: 1.6,
        points: { show: false },
        spanGaps: true,
      });
    }
    return { data, seriesCfg, runIds: series.map(s => s.id) };
  }

  function fmtTooltipNum(v) {
    if (v == null || !isFinite(v)) return '—';
    if (Number.isInteger(v)) return String(v);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(3);
    return String(+v.toPrecision(5));
  }

  function tooltipPlugin() {
    let tip = null;
    return {
      hooks: {
        init: (u) => {
          tip = document.createElement('div');
          tip.className = 'spt-tip';
          tip.style.display = 'none';
          u.over.appendChild(tip);
        },
        destroy: () => {
          if (tip && tip.parentNode) tip.parentNode.removeChild(tip);
          tip = null;
        },
        setCursor: (u) => {
          if (!tip) return;
          const { left, top, idx } = u.cursor;
          if (
            idx == null || left == null || left < 0 ||
            top == null || top < 0
          ) {
            tip.style.display = 'none';
            return;
          }
          const xs = u.data[0];
          if (idx < 0 || idx >= xs.length) {
            tip.style.display = 'none';
            return;
          }
          const xVal = xs[idx];

          const rows = [];
          for (let i = 1; i < u.series.length; i++) {
            const s = u.series[i];
            if (s.show === false) continue;
            const arr = u.data[i];
            if (!arr) continue;
            // Fill-forward: Lightning logs val/ metrics sparsely at different
            // steps than train/, so the exact idx often has nulls. Walk back
            // to the last known value, and if nothing before, forward.
            let kIdx = -1;
            for (let j = Math.min(idx, arr.length - 1); j >= 0; j--) {
              const v = arr[j];
              if (v != null && isFinite(v)) { kIdx = j; break; }
            }
            if (kIdx < 0) {
              for (let j = idx + 1; j < arr.length; j++) {
                const v = arr[j];
                if (v != null && isFinite(v)) { kIdx = j; break; }
              }
            }
            if (kIdx < 0) continue;
            const y = arr[kIdx];
            const color = typeof s.stroke === 'function' ? s.stroke(u, i) : s.stroke;
            rows.push({ color, label: s.runId || s.label || '', value: y });
          }
          if (rows.length === 0) {
            tip.style.display = 'none';
            return;
          }
          rows.sort((a, b) => b.value - a.value);

          // Build content
          tip.replaceChildren();
          const xRow = document.createElement('div');
          xRow.className = 'tt-x';
          xRow.textContent = `x = ${fmtTooltipNum(xVal)}`;
          tip.appendChild(xRow);
          const MAX = 24;
          const shown = rows.slice(0, MAX);
          for (const { color, label, value } of shown) {
            const r = document.createElement('div');
            r.className = 'tt-row';
            const dot = document.createElement('span');
            dot.className = 'tt-dot';
            dot.style.background = color;
            r.appendChild(dot);
            const lab = document.createElement('span');
            lab.className = 'tt-label';
            lab.textContent = label;
            lab.title = label;
            r.appendChild(lab);
            const val = document.createElement('span');
            val.className = 'tt-val';
            val.textContent = fmtTooltipNum(value);
            r.appendChild(val);
            tip.appendChild(r);
          }
          if (rows.length > MAX) {
            const more = document.createElement('div');
            more.className = 'tt-more';
            more.textContent = `+${rows.length - MAX} more`;
            tip.appendChild(more);
          }

          tip.style.display = 'block';
          // Position inside the over element.
          const overW = u.over.clientWidth;
          const overH = u.over.clientHeight;
          const w = tip.offsetWidth;
          const h = tip.offsetHeight;
          let l = left + 14;
          let t = top + 14;
          if (l + w > overW) l = left - w - 14;
          if (t + h > overH) t = top - h - 14;
          if (l < 0) l = 0;
          if (t < 0) t = 0;
          tip.style.transform = `translate(${l}px, ${t}px)`;
        },
      },
    };
  }

  function makeUplotOpts(name, width, seriesCfg) {
    const muted = themeColor('muted') || '#8aa0b8';
    const grid = themeColor('grid') || '#1f2630';
    return {
      width,
      height: 240,
      cursor: {
        drag: { x: true, y: false, uni: 50 },
        // Sync x-cursor across all spt-web charts. Hovering any chart drives
        // the cursor on every other chart at the same x, and each chart's
        // own in-chart tooltip updates with that x's y values.
        sync: { key: SYNC_KEY, setSeries: false },
        focus: { prox: 16 },
      },
      scales: {
        x: { time: false },
        y: { distr: state.logY ? 3 : 1 },
      },
      axes: [
        { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
        { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
      ],
      series: seriesCfg,
      legend: { show: false },
      plugins: [tooltipPlugin()],
    };
  }

  function updateChart(name) {
    const entry = state.charts.get(name);
    if (!entry) return;

    const visibleIds = effectivelyVisible().sort();
    const { data, seriesCfg, runIds } = buildChartData(name, visibleIds);

    if (runIds.length === 0) {
      // No data yet for this chart with current selection; clear.
      if (entry.plot) {
        entry.plot.destroy();
        entry.plot = null;
        entry.configKey = '';
      }
      return;
    }

    const configKey = JSON.stringify([runIds, state.xAxis, state.logY, state.groupBy, state.theme]);

    if (entry.plot && entry.configKey === configKey) {
      entry.plot.setData(data);
      return;
    }

    if (entry.plot) entry.plot.destroy();
    const body = entry.panel.querySelector('.chart-body');
    const width = body.clientWidth || 400;
    entry.plot = new uPlot(makeUplotOpts(name, width, seriesCfg), data, body);
    entry.configKey = configKey;
  }

  // ---- landing / overview ---------------------------------------------

  function timeAgo(epoch) {
    if (!epoch) return '';
    const s = Math.max(0, Date.now() / 1000 - epoch);
    if (s < 60) return `${Math.floor(s)}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  }

  function statCard(label, value, color) {
    const card = document.createElement('div');
    card.className = 'stat-card';
    const v = document.createElement('div');
    v.className = 'stat-value';
    if (color) v.style.color = color;
    v.textContent = String(value);
    const l = document.createElement('div');
    l.className = 'stat-label';
    l.textContent = label;
    card.appendChild(v);
    card.appendChild(l);
    return card;
  }

  function statusBars(counts) {
    const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
    const wrap = document.createElement('div');
    wrap.className = 'status-bars';
    const colors = {
      completed: '#34d399', running: '#22d3ee',
      failed: '#f87171', unknown: '#6b7280',
    };
    const order = ['completed', 'running', 'failed'];
    const seen = new Set(Object.keys(counts));
    const sorted = [
      ...order.filter(k => seen.has(k)),
      ...[...seen].filter(k => !order.includes(k)).sort(),
    ];
    for (const s of sorted) {
      const n = counts[s];
      const pct = (n / total) * 100;
      const row = document.createElement('div');
      row.className = 'status-bar-row';
      const lbl = document.createElement('div');
      lbl.className = 'status-bar-label';
      lbl.textContent = s;
      const track = document.createElement('div');
      track.className = 'status-bar-track';
      const fill = document.createElement('div');
      fill.className = 'status-bar-fill';
      fill.style.width = `${pct}%`;
      fill.style.background = colors[s] || '#6b7280';
      track.appendChild(fill);
      const cnt = document.createElement('div');
      cnt.className = 'status-bar-count';
      cnt.textContent = String(n);
      row.appendChild(lbl);
      row.appendChild(track);
      row.appendChild(cnt);
      wrap.appendChild(row);
    }
    return wrap;
  }

  function tagCloud(runs) {
    const tagCounts = new Map();
    for (const r of runs) {
      for (const t of r.tags || []) {
        tagCounts.set(t, (tagCounts.get(t) || 0) + 1);
      }
    }
    const cloud = document.createElement('div');
    cloud.className = 'tag-cloud';
    if (tagCounts.size === 0) {
      cloud.style.color = '#6b7280';
      cloud.style.fontSize = '12px';
      cloud.textContent = 'no tags';
      return cloud;
    }
    const sorted = [...tagCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 24);
    for (const [t, n] of sorted) {
      const pill = document.createElement('span');
      pill.className = 'tag-pill';
      pill.title = `filter by tag: ${t}`;
      pill.textContent = t;
      const c = document.createElement('span');
      c.className = 'tag-count';
      c.textContent = n;
      pill.appendChild(c);
      pill.addEventListener('click', () => {
        const existing = state.filters.findIndex(f => f.key === 'tags');
        if (existing >= 0) {
          const cur = state.filters[existing];
          if (!cur.values.includes(t)) cur.values.push(t);
        } else {
          state.filters.push({ key: 'tags', values: [t] });
        }
        renderFilters();
        renderRunList();
        scheduleRerender();
      });
      cloud.appendChild(pill);
    }
    return cloud;
  }

  function makeRecentRunItem(r) {
    const row = document.createElement('div');
    row.className = 'recent-run';
    const dot = document.createElement('div');
    dot.className = 'run-dot';
    dot.style.background = runColor(r.run_id);
    row.appendChild(dot);
    const name = document.createElement('div');
    name.className = 'recent-run-name';
    name.textContent = r.run_id;
    name.title = `${r.run_id}\n${r.run_dir || ''}`;
    row.appendChild(name);
    if (r.status) {
      const st = document.createElement('div');
      st.className = `run-status ${r.status}`;
      st.textContent = r.status;
      row.appendChild(st);
    }
    const ago = document.createElement('div');
    ago.className = 'run-ago';
    ago.textContent = timeAgo(r.created_at);
    row.appendChild(ago);
    row.addEventListener('click', () => toggleRun(r.run_id));
    return row;
  }

  function drawActivityTimeline(parent, runs) {
    parent.replaceChildren();
    const now = Math.floor(Date.now() / 1000);
    const oneDay = 86400;
    const days = 30;
    const startDay = Math.floor((now - (days - 1) * oneDay) / oneDay) * oneDay;
    const totalBins = new Array(days).fill(0);
    const failBins = new Array(days).fill(0);
    for (const r of runs) {
      if (!r.created_at) continue;
      const idx = Math.floor((r.created_at - startDay) / oneDay);
      if (idx < 0 || idx >= days) continue;
      totalBins[idx]++;
      if (r.status === 'failed') failBins[idx]++;
    }
    const xs = new Array(days);
    for (let i = 0; i < days; i++) xs[i] = startDay + i * oneDay + oneDay / 2;

    const width = parent.clientWidth || 600;
    const bars = uPlot.paths.bars
      ? uPlot.paths.bars({ size: [0.7, 80], align: 0 })
      : undefined;
    const muted = themeColor('muted') || '#8aa0b8';
    const grid = themeColor('grid') || '#1f2630';
    const accent = themeColor('accent') || '#22d3ee';
    const bad = themeColor('bad') || '#f87171';
    new uPlot({
      width, height: 180,
      cursor: { drag: { x: false, y: false } },
      scales: {
        x: { time: true },
        y: { range: (_, _min, max) => [0, Math.max(1, max + 1)] },
      },
      axes: [
        {
          stroke: muted,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          values: (_, ticks) => ticks.map(t => {
            const d = new Date(t * 1000);
            return `${d.getMonth() + 1}/${d.getDate()}`;
          }),
        },
        {
          stroke: muted,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid },
          size: 40,
        },
      ],
      series: [
        {},
        {
          label: 'runs',
          stroke: accent,
          fill: accent + '59', // ~35% alpha via 8-digit hex
          paths: bars,
          points: { show: false },
        },
        {
          label: 'failed',
          stroke: bad,
          fill: bad + '8c',    // ~55% alpha
          paths: bars,
          points: { show: false },
        },
      ],
      legend: { show: true, live: false },
    }, [xs, totalBins, failBins], parent);
  }

  function renderOverview(root) {
    const allRuns = [...state.runs.values()].filter(passesFilters);
    const counts = {};
    for (const r of allRuns) {
      const s = r.status || 'unknown';
      counts[s] = (counts[s] || 0) + 1;
    }

    const wrap = document.createElement('div');
    wrap.className = 'overview';

    if (allRuns.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.textContent = state.runs.size === 0
        ? 'no runs found in this directory yet'
        : 'no runs match the active filters';
      wrap.appendChild(empty);
      root.replaceChildren(wrap);
      return;
    }

    const cards = document.createElement('div');
    cards.className = 'stat-cards';
    cards.appendChild(statCard('total runs', allRuns.length, '#f1f5f9'));
    cards.appendChild(statCard('running',   counts.running   || 0, '#22d3ee'));
    cards.appendChild(statCard('completed', counts.completed || 0, '#34d399'));
    cards.appendChild(statCard('failed',    counts.failed    || 0, '#f87171'));
    wrap.appendChild(cards);

    const activity = document.createElement('div');
    activity.className = 'overview-card';
    const ah = document.createElement('h3');
    ah.textContent = 'activity (last 30 days)';
    activity.appendChild(ah);
    const ab = document.createElement('div');
    ab.className = 'overview-body';
    activity.appendChild(ab);
    wrap.appendChild(activity);

    const row = document.createElement('div');
    row.className = 'overview-row';

    const recent = document.createElement('div');
    recent.className = 'overview-card';
    const rh = document.createElement('h3');
    rh.textContent = 'recent runs';
    recent.appendChild(rh);
    const rl = document.createElement('div');
    rl.className = 'recent-runs-list';
    const sortedByTime = allRuns.slice()
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      .slice(0, 10);
    for (const r of sortedByTime) rl.appendChild(makeRecentRunItem(r));
    recent.appendChild(rl);
    row.appendChild(recent);

    const sideCol = document.createElement('div');
    sideCol.style.display = 'grid';
    sideCol.style.gap = '12px';
    sideCol.style.alignContent = 'start';

    const status = document.createElement('div');
    status.className = 'overview-card';
    const sh = document.createElement('h3');
    sh.textContent = 'by status';
    status.appendChild(sh);
    status.appendChild(statusBars(counts));
    sideCol.appendChild(status);

    const tags = document.createElement('div');
    tags.className = 'overview-card';
    const th = document.createElement('h3');
    th.textContent = 'tags';
    tags.appendChild(th);
    tags.appendChild(tagCloud(allRuns));
    sideCol.appendChild(tags);

    row.appendChild(sideCol);
    wrap.appendChild(row);

    root.replaceChildren(wrap);

    // Draw the timeline after layout has settled so clientWidth is correct.
    requestAnimationFrame(() => drawActivityTimeline(ab, allRuns));
  }

  // ---- detail (config) modal ------------------------------------------

  function openDetail(runId) {
    state.detailRunId = runId;
    state.detailFilter = '';
    document.getElementById('detail-filter').value = '';
    renderDetail();
    document.getElementById('detail-overlay').hidden = false;
  }

  function closeDetail() {
    state.detailRunId = null;
    document.getElementById('detail-overlay').hidden = true;
  }

  function fmtTime(epoch) {
    if (!epoch) return null;
    const d = new Date(epoch * 1000);
    if (isNaN(d.getTime())) return String(epoch);
    return d.toISOString().replace('T', ' ').replace(/\.\d+Z$/, ' UTC');
  }

  function classifyValue(v) {
    if (v == null) return 'null';
    if (typeof v === 'number') return 'num';
    if (typeof v === 'boolean') return 'bool';
    return '';
  }

  function fmtValue(v) {
    if (v == null) return 'null';
    if (typeof v === 'number') {
      // Compact numeric: keep ints as-is; floats trimmed.
      if (Number.isInteger(v)) return String(v);
      const abs = Math.abs(v);
      if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return v.toExponential(4);
      return String(+v.toPrecision(7));
    }
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) return v.length ? v.join(', ') : '(empty)';
    return String(v);
  }

  function copyToClipboard(text, node) {
    const ok = () => {
      node.classList.add('copied');
      setTimeout(() => node.classList.remove('copied'), 600);
    };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(ok, () => {});
    } else {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); ok(); } catch {}
      ta.remove();
    }
  }

  function buildKVSection(label, kv, filter) {
    const entries = Object.entries(kv)
      .filter(([k]) => !filter || k.toLowerCase().includes(filter))
      .sort((a, b) => a[0].localeCompare(b[0]));
    if (!entries.length) return null;

    const section = document.createElement('section');
    section.className = 'detail-section';
    const h = document.createElement('h3');
    h.textContent = `${label} (${entries.length})`;
    section.appendChild(h);

    const grid = document.createElement('div');
    grid.className = 'kv-grid';
    for (const [k, v] of entries) {
      const dk = document.createElement('div');
      dk.className = 'kv-key';
      dk.textContent = k;
      dk.title = k;
      const dv = document.createElement('div');
      dv.className = 'kv-val ' + classifyValue(v);
      dv.textContent = fmtValue(v);
      const copyText = `${k}=${fmtValue(v)}`;
      dv.title = `click to copy "${copyText}"`;
      dv.addEventListener('click', () => copyToClipboard(copyText, dv));
      grid.appendChild(dk);
      grid.appendChild(dv);
    }
    section.appendChild(grid);
    return section;
  }

  function renderDetail() {
    const r = state.runs.get(state.detailRunId);
    if (!r) return closeDetail();

    document.getElementById('detail-dot').style.background = runColor(r.run_id);
    document.getElementById('detail-title').textContent = r.run_id;

    const meta = {
      run_dir: r.run_dir,
      status: r.status,
      created_at: fmtTime(r.created_at),
      tags: (r.tags || []).join(', ') || null,
      notes: r.notes || null,
      checkpoint_path: r.checkpoint_path,
    };

    const filter = state.detailFilter.toLowerCase();
    const sections = [];
    const metaSec = buildKVSection('meta', meta, filter);
    if (metaSec) sections.push(metaSec);
    const hpSec = buildKVSection('hparams', r.hparams || {}, filter);
    if (hpSec) sections.push(hpSec);
    const smSec = buildKVSection('summary', r.summary || {}, filter);
    if (smSec) sections.push(smSec);

    const body = document.getElementById('detail-body');
    if (!sections.length) {
      const empty = document.createElement('div');
      empty.style.color = '#4a5568';
      empty.style.padding = '16px 0';
      empty.style.textAlign = 'center';
      empty.textContent = filter ? 'no keys match the filter' : 'no config recorded';
      body.replaceChildren(empty);
    } else {
      body.replaceChildren(...sections);
    }
  }

  // ---- SSE --------------------------------------------------------------

  function startSSE() {
    const es = new EventSource('/api/stream');
    es.addEventListener('progress', async (ev) => {
      let p;
      try { p = JSON.parse(ev.data); } catch { return; }
      applyScanStatus(p);
      // The phase flips to 'idle' once the initial scan finishes; that's
      // our cue to fetch the now-populated run list and render the overview.
      if (p && p.initial_done) {
        try { await refreshRuns(); } catch (e) { console.warn(e); }
        scheduleRerender();
      }
    });
    es.addEventListener('update', async (ev) => {
      let payload;
      try { payload = JSON.parse(ev.data); } catch { return; }
      const { changed = [], removed = [] } = payload;

      // Refresh the index (cheap; sidecars only).
      await refreshRuns();

      // Refetch metrics + media for visible-and-changed runs.
      const refetch = changed.filter(id => state.visible.has(id));
      if (refetch.length) {
        await Promise.all([
          ...refetch.map(fetchMetrics),
          ...refetch
            .filter(id => state.runs.get(id)?.has_media)
            .map(fetchMedia),
        ]);
      }

      // Drop removed runs.
      for (const id of removed) {
        state.metrics.delete(id);
        state.media.delete(id);
        state.visible.delete(id);
      }

      if (refetch.length || removed.length || changed.length) {
        scheduleRerender();
      }
      if (state.detailRunId && changed.includes(state.detailRunId)) {
        renderDetail();
      }
      if (state.detailRunId && removed.includes(state.detailRunId)) {
        closeDetail();
      }
    });
    es.addEventListener('error', () => {
      // Browser auto-reconnects.
    });
  }

  // ---- controls ---------------------------------------------------------

  function debounce(fn, ms) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  // ---- image lightbox ---------------------------------------------------

  function openLightbox(src, caption) {
    const box = document.getElementById('image-lightbox');
    const img = document.getElementById('lightbox-img');
    const cap = document.getElementById('lightbox-caption');
    if (!box || !img) return;
    img.src = src;
    img.alt = caption || '';
    cap.textContent = caption || '';
    box.hidden = false;
  }

  function closeLightbox() {
    const box = document.getElementById('image-lightbox');
    const img = document.getElementById('lightbox-img');
    if (!box) return;
    box.hidden = true;
    if (img) img.removeAttribute('src');
  }

  // ---- header stats -----------------------------------------------------

  function updateHeaderStats() {
    const counts = { running: 0, completed: 0, failed: 0 };
    for (const r of state.runs.values()) {
      const s = r.status || 'unknown';
      if (s in counts) counts[s] += 1;
    }
    for (const [k, v] of Object.entries(counts)) {
      const el = document.querySelector(`#stat-${k} .n`);
      if (el) el.textContent = String(v);
    }
  }

  // ---- tabs -------------------------------------------------------------

  function setActiveTab(name) {
    if (!['figures', 'out', 'err'].includes(name)) return;
    state.activeTab = name;
    for (const btn of document.querySelectorAll('#tabs .tab')) {
      btn.classList.toggle('active', btn.dataset.tab === name);
    }
    for (const pane of document.querySelectorAll('.tab-pane')) {
      pane.classList.toggle('active', pane.id === `tab-${name}`);
    }
    if (name === 'out' || name === 'err') {
      // Lazily fetch logs the first time the user opens a log tab.
      refreshLogStreamsForVisibleRuns().then(() => renderLogTab(name));
    }
  }

  // ---- logs -------------------------------------------------------------

  async function fetchLogStreams(runId) {
    try {
      const data = await fetchJSON(`/api/logs?run_id=${encodeURIComponent(runId)}`);
      state.logsIndex.set(runId, { streams: data.streams || [], fetchedAt: Date.now() });
    } catch (e) {
      state.logsIndex.set(runId, { streams: [], fetchedAt: Date.now() });
    }
  }

  async function refreshLogStreamsForVisibleRuns() {
    const ids = effectivelyVisible();
    await Promise.all(ids.map(fetchLogStreams));
  }

  // Build the list of stream options for the active tab as
  // [{value: "<runId>::<stream_id>", label: "<runId> — <stream label>"}, ...]
  function buildLogOptions(kind) {
    const opts = [];
    for (const id of effectivelyVisible()) {
      const idx = state.logsIndex.get(id);
      if (!idx) continue;
      for (const s of idx.streams) {
        if (s.kind !== kind) continue;
        opts.push({
          value: `${id}::${s.stream_id}`,
          label: `${id}  —  ${s.name}  (${formatBytes(s.size)})`,
          runId: id,
          streamId: s.stream_id,
        });
      }
    }
    return opts;
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
  }

  async function renderLogTab(kind) {
    const selectId = kind === 'err' ? 'logs-err-stream-selector' : 'logs-stream-selector';
    const viewId   = kind === 'err' ? 'logs-err' : 'logs-out';
    const sel  = document.getElementById(selectId);
    const view = document.getElementById(viewId);
    const opts = buildLogOptions(kind);

    if (opts.length === 0) {
      sel.innerHTML = '';
      view.classList.add('empty');
      view.textContent = effectivelyVisible().length === 0
        ? 'select one or more runs on the left to view their logs.'
        : `no .${kind} files were found for the selected run(s).`;
      state.logSelection[kind] = null;
      return;
    }

    // Restore prior selection if still valid; else default to the first.
    const prior = state.logSelection[kind];
    const priorVal = prior ? `${prior.runId}::${prior.streamId}` : null;
    let active = opts.find(o => o.value === priorVal) || opts[0];

    sel.innerHTML = '';
    for (const o of opts) {
      const opt = document.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      if (o.value === active.value) opt.selected = true;
      sel.appendChild(opt);
    }
    state.logSelection[kind] = { runId: active.runId, streamId: active.streamId };
    await loadLogContent(kind);
  }

  async function loadLogContent(kind) {
    const sel = state.logSelection[kind];
    const viewId = kind === 'err' ? 'logs-err' : 'logs-out';
    const view = document.getElementById(viewId);
    if (!sel) {
      view.classList.add('empty');
      view.textContent = '';
      return;
    }
    view.classList.remove('empty');
    view.textContent = 'loading...';
    try {
      const r = await fetch(
        `/api/log-content?run_id=${encodeURIComponent(sel.runId)}&stream_id=${encodeURIComponent(sel.streamId)}`,
        { cache: 'no-store' },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const txt = await r.text();
      view.textContent = txt || '(empty)';
      // Auto-scroll to the bottom (most recent output).
      view.scrollTop = view.scrollHeight;
    } catch (e) {
      view.classList.add('empty');
      view.textContent = `failed to load: ${e.message || e}`;
    }
  }

  function wireControls() {
    document.getElementById('run-search').addEventListener(
      'input',
      debounce(e => { state.search = e.target.value; renderRunList(); }, 80)
    );

    const sm = document.getElementById('smoothing');
    const smv = document.getElementById('smoothing-val');
    sm.addEventListener('input', e => {
      state.smoothing = parseFloat(e.target.value);
      smv.textContent = state.smoothing.toFixed(2);
      scheduleRerender();
    });

    document.getElementById('x-axis').addEventListener('change', e => {
      state.xAxis = e.target.value;
      scheduleRerender();
    });

    document.getElementById('log-y').addEventListener('change', e => {
      state.logY = e.target.checked;
      scheduleRerender();
    });

    document.getElementById('select-all').addEventListener('click', () => setAllVisible(true));
    document.getElementById('clear-all').addEventListener('click', () => setAllVisible(false));

    document.getElementById('add-filter-btn').addEventListener('click', () => openFilterDraft(null));

    document.getElementById('group-by').addEventListener('change', e => {
      state.groupBy = e.target.value;
      renderRunList();
      // Group key shows up in chart legends → rebuild charts.
      scheduleRerender();
    });

    // Tabs
    for (const btn of document.querySelectorAll('#tabs .tab')) {
      btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
    }

    // Lightbox: click outside the image (or on the close button) to dismiss.
    const lightbox = document.getElementById('image-lightbox');
    if (lightbox) {
      lightbox.addEventListener('click', e => {
        // Only close when clicking the backdrop, not the image itself.
        if (e.target === lightbox) closeLightbox();
      });
      const closeBtn = document.getElementById('lightbox-close');
      if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
    }
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        const box = document.getElementById('image-lightbox');
        if (box && !box.hidden) closeLightbox();
      }
    });

    // Metric search (debounced; force a tree rebuild because the set of
    // visible metrics changes).
    const ms = document.getElementById('metric-search');
    if (ms) {
      ms.addEventListener('input', debounce(e => {
        state.metricSearch = e.target.value;
        state._lastTreeKey = null; // force rebuild
        scheduleRerender();
      }, 80));
    }

    // Log tab selectors / refresh buttons
    const outSel = document.getElementById('logs-stream-selector');
    if (outSel) outSel.addEventListener('change', () => {
      const [runId, streamId] = outSel.value.split('::');
      state.logSelection.out = { runId, streamId };
      loadLogContent('out');
    });
    const errSel = document.getElementById('logs-err-stream-selector');
    if (errSel) errSel.addEventListener('change', () => {
      const [runId, streamId] = errSel.value.split('::');
      state.logSelection.err = { runId, streamId };
      loadLogContent('err');
    });
    const outRefresh = document.getElementById('logs-refresh');
    if (outRefresh) outRefresh.addEventListener('click', () => loadLogContent('out'));
    const errRefresh = document.getElementById('logs-err-refresh');
    if (errRefresh) errRefresh.addEventListener('click', () => loadLogContent('err'));

    document.getElementById('theme-toggle').addEventListener('click', () => {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
    });

    document.getElementById('sort-by').addEventListener('change', e => {
      state.sortBy = e.target.value;
      renderRunList();
    });

    const sortDirBtn = document.getElementById('sort-dir');
    sortDirBtn.textContent = state.sortDesc ? '↓' : '↑';
    sortDirBtn.addEventListener('click', () => {
      state.sortDesc = !state.sortDesc;
      sortDirBtn.textContent = state.sortDesc ? '↓' : '↑';
      renderRunList();
    });

    // Detail modal wiring.
    document.getElementById('detail-close').addEventListener('click', closeDetail);
    document.getElementById('detail-overlay').addEventListener('click', e => {
      if (e.target.id === 'detail-overlay') closeDetail();
    });
    document.getElementById('detail-filter').addEventListener(
      'input',
      debounce(e => { state.detailFilter = e.target.value; renderDetail(); }, 60)
    );
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && state.detailRunId) closeDetail();
    });

    const onResize = debounce(() => {
      for (const { panel, plot } of state.charts.values()) {
        if (!plot) continue;
        const body = panel.querySelector('.chart-body');
        plot.setSize({ width: body.clientWidth || 400, height: 240 });
      }
    }, 80);
    window.addEventListener('resize', onResize);
  }

  // ---- scan progress overlay -------------------------------------------
  // The backend's initial walk over a large NFS run tree can take many
  // seconds. We render a small overlay so the page is interactive and the
  // user has feedback. The overlay is driven by:
  //   - GET /api/scan-status   (polled once on load)
  //   - SSE 'progress' events  (pushed during the initial scan)
  // When phase === 'idle' (scan finished) we hide the overlay and call
  // refreshRuns() to populate the list.
  function applyScanStatus(p) {
    const overlay = document.getElementById('scan-progress');
    if (!overlay) return;
    if (!p || p.phase === 'idle' || p.initial_done) {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    const fill = document.getElementById('scan-progress-fill');
    const txt  = document.getElementById('scan-progress-text');
    if (p.phase === 'discovering') {
      // The total directory count is unknown during discovery, so we use an
      // indeterminate-but-growing bar. Show the running count of sidecars
      // already found so the user knows it isn't stuck.
      fill.style.width = '5%';
      const found = (p.total || 0).toLocaleString();
      txt.textContent  = `discovering runs (walking directory tree)... ${found} found so far`;
    } else if (p.phase === 'loading') {
      const pct = p.total > 0 ? Math.min(100, (100 * p.done) / p.total) : 0;
      fill.style.width = pct.toFixed(1) + '%';
      txt.textContent  = `loading ${p.done.toLocaleString()} / ${p.total.toLocaleString()} (${pct.toFixed(0)}%)`;
    } else {
      fill.style.width = '100%';
      txt.textContent  = `${p.phase}...`;
    }
  }

  // ---- init -------------------------------------------------------------

  async function main() {
    wireControls();
    initTheme();
    // Start SSE first so we don't miss progress events emitted between the
    // status fetch and the first scan-tick.
    startSSE();
    try {
      const status = await fetchJSON('/api/scan-status');
      applyScanStatus(status);
      if (status && status.initial_done) {
        await refreshRuns();
      }
      // If still scanning, the SSE 'progress' handler will hide the overlay
      // and refresh the run list when phase flips to 'idle'.
    } catch (e) {
      // Older servers without /api/scan-status: fall back to old behaviour.
      console.warn('scan-status unavailable, falling back', e);
      await refreshRuns();
    }
    // Render the stat-card overview immediately. Without this, #charts keeps
    // the static fallback empty-state from the HTML until the user toggles
    // a run on/off — which makes the landing screen look broken.
    scheduleRerender();
  }

  main();
})();
