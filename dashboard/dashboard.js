// PyPTO Dashboard — Interactive
let dashboardData = null;
let currentOp = null;
let currentTab = 'correctness';
let sortField = 'name';
let sortAsc = true;

function init() {
  fetch('./dashboard.json')
    .then(r => r.json())
    .then(data => {
      dashboardData = data;
      renderSummary(data);
      renderTable(data);
      setupSearch();
      setupSort();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('app').style.display = 'block';
    })
    .catch(err => {
      document.getElementById('loading').textContent = 'Failed to load dashboard.json: ' + err.message;
    });
}

function renderSummary(data) {
  const s = data.summary;
  const cards = [
    { label: 'Total Operators', value: s.total, cls: 'color-blue' },
    { label: 'Completed', value: s.completed, cls: 'color-green' },
    { label: 'Torch Ready', value: s.torch_done, cls: 'color-cyan' },
    { label: 'Ascend C Ready', value: s.ascendc_done, cls: 'color-purple' },
    { label: 'PyPTO Ready', value: s.pypto_done, cls: 'color-orange' },
    { label: 'Correctness PASS', value: s.pass_count, cls: 'color-green' },
    { label: 'Correctness FAIL', value: s.fail_count, cls: 'color-red' },
    { label: 'Blocked', value: s.blocked, cls: 'color-yellow' },
  ];
  const container = document.getElementById('summary-cards');
  container.innerHTML = cards.map(c => `
    <div class="summary-card ${c.cls}">
      <div class="value">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');

  // Progress bar
  const pct = s.total > 0 ? Math.round(s.completed / s.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = pct + '% (' + s.completed + '/' + s.total + ')';
  document.getElementById('update-time').textContent = 'Generated: ' + data.generated_at;
}

function renderTable(data) {
  const ops = [...data.operators];
  ops.sort((a, b) => {
    let va = String(a[sortField] || '').toLowerCase();
    let vb = String(b[sortField] || '').toLowerCase();
    if (sortField === 'name') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    // For status
    if (sortField === 'status') {
      const order = { completed: 0, in_progress: 1, planned: 2, blocked: 3 };
      const da = order[a.status] || 99;
      const db = order[b.status] || 99;
      return sortAsc ? da - db : db - da;
    }
    return 0;
  });

  const tbody = document.getElementById('op-table-body');
  tbody.innerHTML = ops.map(op => {
    const corr = op.correctness_all_pass === true ? '<span class="status-badge pass">PASS</span>' :
                 op.correctness_all_pass === false ? '<span class="status-badge fail">FAIL</span>' :
                 '<span class="status-badge unknown">N/A</span>';

    const torchKt = op.kernel_types?.torch?.join(', ') || 'N/A';
    const ascendcKt = op.kernel_types?.ascendc?.join(', ') || 'N/A';
    const pyptoKt = op.kernel_types?.pypto?.join(', ') || 'N/A';

    const ktc = op.kernel_count || {};
    const kcTorch = ktc.torch ?? 'N/A';
    const kcAscendc = ktc.ascendc ?? 'N/A';
    const kcPypto = ktc.pypto ?? 'N/A';

    // Benchmark table data
    let tb = op.torch?.benchmark;
    let ab = op.ascendc?.benchmark;
    let pb = op.pypto?.benchmark;
    let perf1 = '';
    if (op.batches && op.batches.length > 0) {
      const b1 = String(op.batches[0]);
      const tLat = tb?.[b1]?.median_us != null ? tb[b1].median_us.toFixed(1) + 'μs' : '';
      const aLat = ab?.[b1]?.median_us != null ? ab[b1].median_us.toFixed(1) + 'μs' : '';
      const pLat = pb?.[b1]?.median_us != null ? pb[b1].median_us.toFixed(1) + 'μs' : '';
      // Try profiler data
      const pd = op.profiler_data?.[b1];
      if (pd) {
        const tf = pd.torch?.primary_compute_kernel_us;
        const af = pd.ascendc?.primary_compute_kernel_us;
        const pf = pd.pypto?.primary_compute_kernel_us;
        if (tf != null) perf1 += 'T:' + tf.toFixed(1) + 'μs ';
        if (af != null) perf1 += 'A:' + af.toFixed(1) + 'μs ';
        if (pf != null) perf1 += 'P:' + pf.toFixed(1) + 'μs';
      } else {
        if (tLat) perf1 += 'T:' + tLat + ' ';
        if (aLat) perf1 += 'A:' + aLat + ' ';
        if (pLat) perf1 += 'P:' + pLat;
      }
    } else if (op.comparison_summary && op.comparison_summary[0]) {
      const s0 = op.comparison_summary[0];
      const tf = s0.torch_primary_us || s0.torch_us;
      const af = s0.ascendc_primary_us || s0.ascendc_us;
      const pf = s0.pypto_primary_us || s0.pypto_compute_us;
      if (tf != null) perf1 += 'T:' + Number(tf).toFixed(1) + 'μs ';
      if (af != null) perf1 += 'A:' + Number(af).toFixed(1) + 'μs ';
      if (pf != null) perf1 += 'P:' + Number(pf).toFixed(1) + 'μs';
    }
    if (!perf1) perf1 = 'N/A';

    return `<tr onclick="showDetail('${op.name}')">
      <td><strong>${op.name}</strong></td>
      <td><span class="status-badge ${op.status}">${op.status}</span></td>
      <td>${torchKt}</td>
      <td>${ascendcKt}</td>
      <td>${pyptoKt}</td>
      <td>${corr}</td>
      <td style="font-size:12px">${perf1}</td>
      <td style="font-size:12px">${op.last_update}</td>
    </tr>`;
  }).join('');

  // Update sort arrows
  document.querySelectorAll('#op-table th').forEach(th => {
    const field = th.dataset.sort;
    th.classList.toggle('sorted', field === sortField);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow && field === sortField) {
      arrow.textContent = sortAsc ? ' ▲' : ' ▼';
    }
  });
}

function setupSearch() {
  document.getElementById('search').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#op-table-body tr');
    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(q) ? '' : 'none';
    });
  });
}

function setupSort() {
  document.querySelectorAll('#op-table th').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (!field) return;
      if (sortField === field) {
        sortAsc = !sortAsc;
      } else {
        sortField = field;
        sortAsc = true;
      }
      renderTable(dashboardData);
    });
  });
}

function showDetail(opName) {
  const op = dashboardData.operators.find(o => o.name === opName);
  if (!op) return;
  currentOp = op;
  currentTab = 'correctness';

  document.getElementById('detail-view').classList.add('visible');
  document.getElementById('detail-title').textContent = op.name + ' — Operator Detail';
  renderBasicInfo(op);
  renderDevStatus(op);
  renderCorrectness(op);
  renderPerformance(op);
  renderComparison(op);
  renderKernelTimeline(op);
  renderKernelTypeChart(op);
  renderProfiler(op);
  renderHistory(op);

  // Activate default tab
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('[data-tab="correctness"]').classList.add('active');
  document.getElementById('tab-correctness').classList.add('active');

  document.getElementById('detail-view').scrollIntoView({ behavior: 'smooth' });
}

function closeDetail() {
  document.getElementById('detail-view').classList.remove('visible');
  currentOp = null;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`tab-${tab}`).classList.add('active');
}

function kernelTag(type) {
  if (!type) return '<span class="kernel-tag default">N/A</span>';
  const t = type.toUpperCase();
  if (t.includes('AIVEC')) return '<span class="kernel-tag aivec">AIVEC</span>';
  if (t.includes('MIX_AIC') || t.includes('MIX') || t === 'MIX_AIC') return '<span class="kernel-tag mix">MIX_AIC</span>';
  if (t.includes('AICPU')) return '<span class="kernel-tag aicpu">AICPU</span>';
  if (t.includes('AIC')) return '<span class="kernel-tag aic">AIC</span>';
  return `<span class="kernel-tag default">${type}</span>`;
}

function renderBasicInfo(op) {
  document.getElementById('info-formula').textContent = op.formula || 'N/A';
  document.getElementById('info-shape').textContent = op.shape || 'N/A';
  document.getElementById('info-dtype').textContent = op.dtype || 'N/A';
  document.getElementById('info-batches').textContent = (op.batches || []).join(', ') || 'N/A';
  document.getElementById('info-broadcast').textContent = op.broadcast ? 'Yes' : 'No';
  document.getElementById('info-kernel-shape').textContent = op.kernel_shape || 'N/A';
  document.getElementById('info-logical-shape').textContent = op.logical_shape || 'N/A';
  document.getElementById('info-precision').textContent = op.precision || 'N/A';
  document.getElementById('info-fastest').textContent = op.fastest || 'N/A';
  document.getElementById('info-status').innerHTML = `<span class="status-badge ${op.status}">${op.status}</span>`;
}

function renderDevStatus(op) {
  const sd = op.dev_status_detail || {};
  const steps = ['Intent', 'API', 'Golden', 'Design', 'Implementation', 'Correctness', 'Benchmark', 'Archive'];
  const html = steps.map(s => {
    const st = sd[s] || 'pending';
    return `<span class="step ${st}">${st === 'completed' ? '✓' : st === 'failed' ? '✗' : '○'} ${s}</span>`;
  }).join('<span class="arrow">→</span>');
  document.getElementById('dev-pipeline').innerHTML = html;
}

function renderCorrectness(op) {
  const corr = op.correctness || {};
  const allPass = op.correctness_all_pass;
  const statusHtml = allPass === true ? '<span class="status-badge pass" style="font-size:16px;padding:4px 16px">ALL PASS</span>' :
                     allPass === false ? '<span class="status-badge fail" style="font-size:16px;padding:4px 16px">FAIL</span>' :
                     '<span class="status-badge unknown" style="font-size:16px;padding:4px 16px">UNKNOWN</span>';
  document.getElementById('corr-status').innerHTML = statusHtml;

  // Per-batch per-impl table
  const batches = op.batches || [];
  if (batches.length === 0) {
    document.getElementById('corr-table-body').innerHTML = '<tr><td colspan="6">No correctness data</td></tr>';
    return;
  }

  const impls = ['torch', 'ascendc', 'pypto'];
  let html = '';
  for (const bk of batches) {
    const sk = String(bk);
    html += `<tr><td>B=${bk}</td>`;
    for (const impl of impls) {
      const c = corr[impl]?.[sk];
      const s = c?.status || 'N/A';
      const cls = s === 'PASS' ? 'pass' : s === 'FAIL' ? 'fail' : 'unknown';
      const md = c?.max_abs_diff != null ? Number(c.max_abs_diff).toExponential(2) : '-';
      html += `<td><span class="status-badge ${cls}">${s}</span></td>
               <td style="font-size:12px">md=${md}</td>`;
    }
    html += '</tr>';
  }
  document.getElementById('corr-table-body').innerHTML = html;

  // Heatmap
  const heatmapContainer = document.getElementById('corr-heatmap');
  let hm = '<div class="heatmap" style="grid-template-columns:repeat(' + impls.length + ', 28px)">';
  for (const impl of impls) {
    hm += `<div style="font-size:10px;color:var(--text-muted);text-align:center">${impl[0].toUpperCase() + impl.slice(1,3)}</div>`;
  }
  for (const bk of batches) {
    const sk = String(bk);
    for (const impl of impls) {
      const c = corr[impl]?.[sk];
      const s = c?.status || 'N/A';
      const cls = s === 'PASS' ? 'pass' : s === 'FAIL' ? 'fail' : 'na';
      hm += `<div class="heatmap-cell ${cls}">${s === 'PASS' ? '✓' : s === 'FAIL' ? '✗' : '?'}</div>`;
    }
  }
  hm += '</div>';
  heatmapContainer.innerHTML = hm;

  // Note
  document.getElementById('corr-note').textContent = corr.note || '';
}

function renderPerformance(op) {
  const pdata = op.profiler_data || {};
  const summary = op.comparison_summary || [];
  const batches = op.batches || [];
  const container = document.getElementById('perf-table-body');

  if (summary.length > 0) {
    let html = '';
    for (const s of summary) {
      const bk = s.batch;
      const t = s.torch_primary_us ?? s.torch_us ?? 'N/A';
      const a = s.ascendc_primary_us ?? s.ascendc_us ?? 'N/A';
      const pc = s.pypto_primary_us ?? s.pypto_compute_us ?? 'N/A';
      const pt = s.pypto_total_dev_us ?? s.pypto_total_us ?? 'N/A';
      const f = s.fastest || '';
      const tStr = t !== 'N/A' ? Number(t).toFixed(1) : 'N/A';
      const aStr = a !== 'N/A' ? Number(a).toFixed(1) : 'N/A';
      const pcStr = pc !== 'N/A' ? Number(pc).toFixed(1) : 'N/A';
      const ptStr = pt !== 'N/A' ? Number(pt).toFixed(1) : 'N/A';
      html += `<tr>
        <td>B=${bk}</td>
        <td class="${f === 'torch' ? 'fastest' : ''}">${tStr} μs</td>
        <td class="${f === 'ascendc' ? 'fastest' : ''}">${aStr} μs</td>
        <td class="${f === 'pypto' ? 'fastest' : ''}">${pcStr} μs</td>
        <td>${ptStr} μs</td>
      </tr>`;
    }
    container.innerHTML = html;
  } else if (Object.keys(pdata).length > 0) {
    let html = '';
    for (const [bk, bv] of Object.entries(pdata)) {
      const t = bv.torch?.primary_compute_kernel_us;
      const a = bv.ascendc?.primary_compute_kernel_us;
      const p = bv.pypto?.primary_compute_kernel_us;
      const pt = bv.pypto?.all_device_kernels_per_call_us ?? bv.pypto?.all_device_kernels_us;
      html += `<tr>
        <td>B=${bk}</td>
        <td>${t != null ? t.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${a != null ? a.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${p != null ? p.toFixed(1) + ' μs' : 'N/A'}</td>
        <td>${pt != null ? pt.toFixed(1) + ' μs' : 'N/A'}</td>
      </tr>`;
    }
    container.innerHTML = html;
  } else {
    // Try benchmark data
    const impls = ['torch', 'ascendc', 'pypto'];
    let html = '';
    for (const bk of batches) {
      const sk = String(bk);
      html += `<tr><td>B=${bk}</td>`;
      for (const impl of impls) {
        const bm = op[impl]?.benchmark?.[sk];
        const v = bm?.median_us;
        html += `<td>${v != null ? v.toFixed(1) + ' μs' : 'N/A'}</td>`;
      }
      html += '</tr>';
    }
    container.innerHTML = html || '<tr><td colspan="5">No performance data</td></tr>';
  }
}

function renderComparison(op) {
  const summary = op.comparison_summary || [];
  const pdata = op.profiler_data || {};
  const container = document.getElementById('comparison-body');
  const batches = op.batches || [];

  if (summary.length > 0) {
    const ref = summary[0];
    cellL = document.getElementById('comp-ktype-torch');
    cellL.textContent = ref.torch_kernel_type || 'N/A';
    cellL = document.getElementById('comp-ktype-ascendc');
    cellL.textContent = ref.ascendc_kernel_type || 'N/A';
    cellL = document.getElementById('comp-ktype-pypto');
    cellL.textContent = ref.pypto_primary_type || 'N/A';

    const kc = op.kernel_count || {};
    document.getElementById('comp-kcount-torch').textContent = kc.torch ?? 'N/A';
    document.getElementById('comp-kcount-ascendc').textContent = kc.ascendc ?? 'N/A';
    document.getElementById('comp-kcount-pypto').textContent = kc.pypto ?? 'N/A';

    const latencies = ['torch_primary_us', 'ascendc_primary_us', 'pypto_primary_us'].map(k => {
      const v = ref[k] || ref[k.replace('_primary_us', '_us')];
      return v != null ? Number(v).toFixed(1) + ' μs' : 'N/A';
    });
    document.getElementById('comp-latency-torch').textContent = latencies[0];
    document.getElementById('comp-latency-ascendc').textContent = latencies[1];
    document.getElementById('comp-latency-pypto').textContent = latencies[2];

    // BW, GFLOPS from summary
    const bw = [ref.torch_bw_gbs, ref.ascendc_bw_gbs, ref.pypto_bw_gbs];
    document.getElementById('comp-bw-torch').textContent = bw[0] != null ? bw[0].toFixed(1) + ' GB/s' : 'N/A';
    document.getElementById('comp-bw-ascendc').textContent = bw[1] != null ? bw[1].toFixed(1) + ' GB/s' : 'N/A';
    document.getElementById('comp-bw-pypto').textContent = bw[2] != null ? bw[2].toFixed(1) + ' GB/s' : 'N/A';
  } else {
    // Try to get from pdata
    const firstKey = Object.keys(pdata)[0];
    if (firstKey) {
      const bv = pdata[firstKey];
      document.getElementById('comp-ktype-torch').textContent = bv.torch?.kernel_type || 'N/A';
      document.getElementById('comp-ktype-ascendc').textContent = bv.ascendc?.kernel_type || 'N/A';
      document.getElementById('comp-ktype-pypto').textContent = bv.pypto?.primary_kernel_type || bv.pypto?.kernel_type || 'N/A';

      const kc = op.kernel_count || {};
      document.getElementById('comp-kcount-torch').textContent = kc.torch ?? bv.torch?.kernels_per_call ?? 'N/A';
      document.getElementById('comp-kcount-ascendc').textContent = kc.ascendc ?? bv.ascendc?.kernels_per_call ?? 'N/A';
      document.getElementById('comp-kcount-pypto').textContent = kc.pypto ?? 'N/A';

      const getLat = (d) => d?.primary_compute_kernel_us != null ? d.primary_compute_kernel_us.toFixed(1) + ' μs' : 'N/A';
      document.getElementById('comp-latency-torch').textContent = getLat(bv.torch);
      document.getElementById('comp-latency-ascendc').textContent = getLat(bv.ascendc);
      document.getElementById('comp-latency-pypto').textContent = getLat(bv.pypto);
    }
  }

  // Correctness
  const allPass = op.correctness_all_pass;
  const corrStatus = allPass === true ? 'PASS' : allPass === false ? 'FAIL' : 'UNKNOWN';
  document.getElementById('comp-corr-torch').textContent = corrStatus;
  document.getElementById('comp-corr-ascendc').textContent = corrStatus;
  document.getElementById('comp-corr-pypto').textContent = corrStatus;
}

function renderKernelTimeline(op) {
  const pdata = op.profiler_data || {};
  const firstKey = Object.keys(pdata)[0];
  const container = document.getElementById('kernel-timeline');

  if (!firstKey) {
    container.innerHTML = '<div class="section"><h3>Kernel Timeline</h3><p class="text-secondary">No profiler data</p></div>';
    return;
  }

  const bv = pdata[firstKey];
  const tk = bv.torch?.primary_compute_kernel_us || 0;
  const ak = bv.ascendc?.primary_compute_kernel_us || 0;
  const pk = bv.pypto?.primary_compute_kernel_us || 0;
  const pt = bv.pypto?.all_device_kernels_per_call_us || bv.pypto?.all_device_kernels_us || 0;

  const maxDur = Math.max(tk, ak, pt, 1);
  const scale = 400 / maxDur;

  const tBar = (tk * scale).toFixed(0);
  const aBar = (ak * scale).toFixed(0);
  const pBar = (pk * scale).toFixed(0);
  const ptBar = (pt * scale).toFixed(0);

  const tn = bv.torch?.kernel_names?.[0] || 'torch_kernel';
  const an = bv.ascendc?.kernel_names?.[0] || 'ascendc_kernel';
  const pn = bv.pypto?.kernel_names?.[0] || 'pypto_kernel';

  container.innerHTML = `
    <div class="section">
      <h3>Kernel Timeline <span style="font-size:12px;color:var(--text-muted)">(B=${firstKey})</span></h3>
      <div class="kernel-timeline">
        <div class="kernel-row">
          <div class="kernel-label">Torch</div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar torch" style="width:${tBar}px"><span class="bar-label">${tn}</span></div>
          </div>
          <div class="kernel-dur">${tk.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">Ascend C</div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar ascendc" style="width:${aBar}px"><span class="bar-label">${an}</span></div>
          </div>
          <div class="kernel-dur">${ak.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">PyPTO<br><span style="font-size:11px;color:var(--text-muted)">compute</span></div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar pypto" style="width:${pBar}px"><span class="bar-label">${pn}</span></div>
          </div>
          <div class="kernel-dur">${pk.toFixed(1)} μs</div>
        </div>
        <div class="kernel-row">
          <div class="kernel-label">PyPTO<br><span style="font-size:11px;color:var(--text-muted)">total</span></div>
          <div class="kernel-bar-wrapper">
            <div class="kernel-bar pypto" style="width:${ptBar}px"><span class="bar-label">+ executor</span></div>
          </div>
          <div class="kernel-dur">${pt.toFixed(1)} μs</div>
        </div>
      </div>
    </div>`;
}

function renderKernelTypeChart(op) {
  const container = document.getElementById('kernel-type-chart');
  const types = op.kernel_types || {};
  const count = op.kernel_count || {};

  const allTypes = {};
  for (const [impl, kts] of Object.entries(types)) {
    for (const kt of (Array.isArray(kts) ? kts : [kts])) {
      if (kt && kt !== 'N/A') {
        allTypes[kt] = (allTypes[kt] || 0) + 1;
      }
    }
  }

  if (Object.keys(allTypes).length === 0) {
    container.innerHTML = '<p class="text-secondary">No kernel type data</p>';
    return;
  }

  const colors = {
    'KERNEL_AIVEC': '#58a6ff',
    'KERNEL_MIX_AIC': '#bc8cff',
    'KERNEL_AICPU': '#f0883e',
    'KERNEL_AIC': '#39d2c0',
  };

  const total = Object.values(allTypes).reduce((a, b) => a + b, 0);
  let html = '<div style="display:flex;align-items:center;gap:24px"><div style="position:relative;width:200px;height:200px">';
  html += '<canvas id="kernel-pie" width="200" height="200"></canvas></div><div>';

  // Legend
  html += '<div class="legend">';
  for (const [kt, cnt] of Object.entries(allTypes)) {
    const pct = Math.round(cnt / total * 100);
    const c = colors[kt] || '#6e7681';
    html += `<div class="legend-item"><span class="dot" style="background:${c}"></span>${kt}: ${cnt} (${pct}%)</div>`;
  }
  html += '</div></div></div>';
  container.innerHTML = html;

  // Draw pie
  setTimeout(() => {
    const canvas = document.getElementById('kernel-pie');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const entries = Object.entries(allTypes);
    let startAngle = -Math.PI / 2;
    entries.forEach(([kt, cnt]) => {
      const sliceAngle = (cnt / total) * 2 * Math.PI;
      const c = colors[kt] || '#6e7681';
      ctx.beginPath();
      ctx.moveTo(100, 100);
      ctx.arc(100, 100, 90, startAngle, startAngle + sliceAngle);
      ctx.closePath();
      ctx.fillStyle = c;
      ctx.fill();
      startAngle += sliceAngle;
    });
  }, 50);
}

function renderProfiler(op) {
  const parsed = op.parsed || {};
  const container = document.getElementById('profiler-body');

  const entries = Object.entries(parsed).slice(0, 5);
  if (entries.length === 0) {
    container.innerHTML = '<tr><td colspan="7">No profiler data</td></tr>';
    return;
  }

  let html = '';
  for (const [key, pdata] of entries) {
    const events = pdata.kernel?.kernel_events || [];
    const byType = pdata.kernel?.by_type || {};
    const byName = pdata.kernel?.by_name || {};
    const totalDur = pdata.kernel?.total_kernel_dur_us || 0;
    const kernelCount = pdata.kernel?.kernel_count || 0;

    for (const [name, info] of Object.entries(byName)) {
      html += `<tr>
        <td style="font-size:12px">${key}</td>
        <td style="font-size:12px">${name}</td>
        <td>${info.mean_dur_us != null ? info.mean_dur_us.toFixed(2) + ' μs' : 'N/A'}</td>
        <td>${info.count}</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>N/A</td>
      </tr>`;
    }

    for (const [type, info] of Object.entries(byType)) {
      html += `<tr>
        <td style="font-size:12px">${key}</td>
        <td style="font-size:12px"><span class="kernel-tag ${type.toLowerCase().includes('aivec') ? 'aivec' : type.toLowerCase().includes('aicpu') ? 'aicpu' : type.toLowerCase().includes('mix') ? 'mix' : 'default'}">${type}</span></td>
        <td>${info.mean_dur_us != null ? info.mean_dur_us.toFixed(2) + ' μs' : 'N/A'}</td>
        <td>${info.count}</td>
        <td>N/A</td>
        <td>N/A</td>
        <td>N/A</td>
      </tr>`;
    }
  }
  container.innerHTML = html;
}

function renderHistory(op) {
  const history = op.history || [];
  const container = document.getElementById('history-content');

  if (history.length === 0) {
    container.innerHTML = '<p class="text-secondary">No history data</p>';
    return;
  }

  let html = '';
  for (const h of history) {
    const hd = h.data;
    const corr = hd.correctness?.all_batches_pass !== undefined ? (hd.correctness.all_batches_pass ? 'PASS' : 'FAIL') : 'N/A';
    const perfStr = hd.comparison_summary?.[0] ? 'B1 torch=' + hd.comparison_summary[0].torch_primary_us?.toFixed(1) + 'μs' : 'N/A';
    html += `<div class="history-item">
      <div class="version">${h.version}</div>
      <div style="display:flex;gap:24px;margin-top:4px;font-size:13px">
        <span>Correctness: <span class="status-badge ${corr === 'PASS' ? 'pass' : 'fail'}">${corr}</span></span>
        <span>Latency: ${perfStr}</span>
      </div>
    </div>`;
  }
  container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', init);
