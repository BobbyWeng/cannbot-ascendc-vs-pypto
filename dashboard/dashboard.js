const ROUTES = ['torch', 'ascendc', 'pypto'];
const ROUTE_NAMES = { torch: 'Torch', ascendc: 'Ascend C', pypto: 'PyPTO' };
let state = { data: null, selected: null, query: '', status: 'all' };

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[char]));
const fmt = value => value == null ? 'N/A' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 3 });
const statusClass = status => status === 'COMPLETE' ? 'complete' : status === 'COMPLETE_WITH_LIMITATION' ? 'limited' : 'limited';
const resultClass = status => ({ PASS: 'pass', PARTIAL: 'partial', FAIL: 'fail' }[status] || 'na');

async function loadDashboard() {
  const response = await fetch('./dashboard.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  if (data.schema_version !== '3.0') throw new Error(`不支持的数据模型：${data.schema_version || 'unknown'}`);
  return data;
}

function evidenceQuality(operator) {
  const stats = operator.performance.evidence_stats;
  if (stats.mismatch) return { label: `${stats.mismatch} 条哈希冲突`, cls: 'mismatch' };
  if (stats.unmanifested) return { label: `${stats.unmanifested} 条未入清单`, cls: 'unmanifested' };
  if (stats.release_source) return { label: '发布源 / SHA 已验证', cls: 'release' };
  if (stats.verified) return { label: `${stats.verified} 条 SHA 已验证`, cls: 'verified' };
  return { label: '无性能证据', cls: 'na' };
}

function renderHeader(data) {
  $('#release-meta').textContent = `版本 ${data.release_version} · 发布于 ${data.generated_at} · 源提交 ${data.release_git_commit.slice(0, 10)}`;
  const evidence = data.summary.evidence;
  const health = $('#source-health');
  health.textContent = evidence.mismatch === 0 ? '数据链无哈希冲突' : `${evidence.mismatch} 条哈希冲突`;
  health.className = `pill ${evidence.mismatch === 0 ? 'good' : 'warn'}`;
  $('#source-alert').innerHTML = `<span>ⓘ</span><div><strong>数据口径：</strong>状态与正确性只来自 <code>${esc(data.source.release_file)}</code>；性能优先采用 release 结构化 profiler，其次采用 SHA256 可追溯的 parsed profiler。旧 performance_matrix 未参与生成，未入清单的数据只展示、不排名。</div>`;
}

function renderSummary(data) {
  const s = data.summary;
  const evidenceCount = s.evidence.release_source + s.evidence.verified;
  const cards = [
    ['算子总数', s.operators, 'blue', '当前发布覆盖'],
    ['完成', `${s.complete}/${s.operators}`, 'green', '含有限制完成'],
    ['路线正确性 PASS', `${s.correctness_pass_routes}/${s.total_routes}`, 'cyan', '以 current_release 为准'],
    ['已校验性能证据', evidenceCount, 'purple', 'release + SHA verified'],
    ['待审计记录', s.evidence.unmanifested, 'yellow', '展示但不参与排名'],
  ];
  $('#summary').innerHTML = cards.map(([label, value, cls, note]) => `
    <article class="summary-card ${cls}">
      <div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="note">${esc(note)}</div>
    </article>`).join('');
}

function updateTitle(key) {
  const names = {
    fix_matmul_multicore_batch: 'MatMul 多核批调度',
    fix_reduce_sum_fp32_kernel: 'ReduceSum FP32 累积',
    fix_add_correctness_reference: 'Add 正确性基准',
    fix_dashboard_chinese: '中文 Dashboard v3',
  };
  return names[key] || key.replaceAll('_', ' ');
}

function renderUpdates(data) {
  const entries = Object.entries(data.post_rc3_fixes || {});
  $('#updates').innerHTML = entries.map(([key, value], index) => `
    <article class="update-card"><span class="index">0${index + 1}</span><h3>${esc(updateTitle(key))}</h3><p>${esc(value)}</p></article>
  `).join('') || '<article class="update-card"><p>发布源没有记录更新摘要。</p></article>';
}

function b1Latency(operator) {
  const b1 = operator.performance.batches['1'] || {};
  const items = ROUTES.filter(route => b1[route]?.primary_us != null && b1[route]?.comparable === true).map(route =>
    `<span class="route-${route}">${ROUTE_NAMES[route]} ${fmt(b1[route].primary_us)} µs</span>`
  );
  return items.length ? items.join('') : '<span class="muted">N/A</span>';
}

function correctnessCell(result) {
  return `<div class="route-cell"><span class="result ${resultClass(result.status)}">${esc(result.status)}</span><span class="route-detail" title="${esc(result.detail)}">${esc(result.detail)}</span></div>`;
}

function renderOperators() {
  const operators = Object.values(state.data.operators).filter(operator => {
    const text = JSON.stringify(operator).toLowerCase();
    return (state.status === 'all' || operator.status === state.status) && text.includes(state.query.toLowerCase());
  }).sort((a, b) => {
    const order = { COMPLETE: 0, COMPLETE_WITH_LIMITATION: 1, PARTIAL: 2, INCOMPLETE: 3 };
    return (order[a.status] ?? 9) - (order[b.status] ?? 9) || a.name.localeCompare(b.name);
  });

  $('#operator-rows').innerHTML = operators.map(operator => {
    const quality = evidenceQuality(operator);
    return `<tr data-operator="${esc(operator.name)}" tabindex="0">
      <td><span class="operator-name">${esc(operator.name)}</span><span class="operator-sub" title="${esc(operator.formula)}">${esc(operator.formula)}</span></td>
      <td><span class="status ${statusClass(operator.status)}">${esc(operator.status_zh)}</span></td>
      <td>${correctnessCell(operator.correctness.torch)}</td>
      <td>${correctnessCell(operator.correctness.ascendc)}</td>
      <td>${correctnessCell(operator.correctness.pypto)}</td>
      <td><div class="latency-stack">${b1Latency(operator)}</div></td>
      <td><span class="quality ${quality.cls}">${esc(quality.label)}</span></td>
      <td class="arrow">›</td>
    </tr>`;
  }).join('');
  $('#empty-state').hidden = operators.length > 0;
  $$('#operator-rows tr').forEach(row => {
    const open = () => showDetail(row.dataset.operator);
    row.addEventListener('click', open);
    row.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') open(); });
  });
}

function info(label, value) {
  return `<div class="info"><div class="label">${esc(label)}</div><div class="value">${esc(value || 'N/A')}</div></div>`;
}

function renderOverview(operator) {
  const env = state.data.environment || {};
  $('#tab-overview').innerHTML = `
    <div class="info-grid">
      ${info('Shape', operator.shape)}${info('Dtype', operator.dtype)}${info('Batches', operator.batches.join(', '))}${info('精度标准', operator.precision)}
      ${info('CANN', env.cann || env.cann_version)}${info('PyTorch', env.torch || env.pytorch)}${info('torch_npu', env.torch_npu)}${info('设备', env.device || env.npu_model)}
    </div>
    <div class="section"><h3>发布覆盖声明</h3><div class="evidence-list">
      <div class="evidence-item"><strong>正确性</strong><p>${esc(operator.correctness_coverage)}</p></div>
      <div class="evidence-item"><strong>Profiler</strong><p>${esc(operator.profiler_coverage)}</p></div>
    </div></div>`;
}

function renderCorrectness(operator) {
  const cards = ROUTES.map(route => {
    const result = operator.correctness[route];
    return `<article class="correctness-card"><h3>${ROUTE_NAMES[route]}</h3><span class="result ${resultClass(result.status)}">${esc(result.status)}</span><p>${esc(result.detail)}</p></article>`;
  }).join('');
  $('#tab-correctness').innerHTML = `<div class="correctness-grid">${cards}</div>
    <div class="section"><h3>门禁说明</h3><p class="section-note">性能“最快”仅在当前路线 correctness=PASS、数据来自 release 或 SHA 已验证、且测量方法为 msprof 时计算。其他数据仍可查看，但明确显示“不排名”。</p></div>`;
}

function rankable(operator, route, record) {
  return operator.correctness[route].status === 'PASS' && record?.comparable === true &&
    ['RELEASE_SOURCE', 'VERIFIED'].includes(record.integrity) && String(record.method).toLowerCase().includes('msprof') && record.primary_us != null;
}

function renderPerformance(operator) {
  const batches = Object.keys(operator.performance.batches).sort((a, b) => Number(a) - Number(b));
  const comparisonRows = batches.map(batch => {
    const records = operator.performance.batches[batch];
    const eligible = ROUTES.filter(route => rankable(operator, route, records[route]));
    let fastest = null;
    if (eligible.length >= 2) fastest = eligible.reduce((best, route) => records[route].primary_us < records[best].primary_us ? route : best);
    const values = ROUTES.map(route => {
      const record = records[route];
      const cls = fastest === route ? 'fastest' : '';
      return `<td class="${cls}">${record?.primary_us == null ? 'N/A' : `${fmt(record.primary_us)} µs`}</td>`;
    }).join('');
    return `<tr><td>B=${esc(batch)}</td>${values}<td>${fastest ? `<span class="fastest">${ROUTE_NAMES[fastest]}</span>` : '<span class="not-ranked">不排名</span>'}</td></tr>`;
  }).join('');

  const detailRows = batches.flatMap(batch => ROUTES.map(route => {
    const record = operator.performance.batches[batch][route];
    if (!record) return '';
    const qualityClass = record.integrity === 'UNMANIFESTED' ? 'unmanifested' : record.integrity === 'RELEASE_SOURCE' ? 'release' : 'verified';
    const note = record.comparison_note ? `<br><span class="not-ranked">${esc(record.comparison_note)}</span>` : '';
    return `<tr><td>B=${esc(batch)}</td><td class="route-${route}">${ROUTE_NAMES[route]}</td>
      <td>${fmt(record.primary_us)}</td><td>${fmt(record.all_device_us)}</td><td>${fmt(record.executor_us)}</td><td>${fmt(record.tflops)}</td>
      <td>${esc(record.kernel_type)}</td><td>${esc(record.kernels_per_call ?? 'N/A')}</td><td>${esc(record.block_dim ?? 'N/A')}</td>
      <td><span class="quality ${qualityClass}">${esc(record.integrity)}</span>${note}</td></tr>`;
  })).join('');

  $('#tab-performance').innerHTML = `
    <div class="section"><h3>主计算核延迟对比</h3><p class="section-note">统一指标：primary_compute_kernel_us。不同测量方法、未入 SHA 清单或未通过正确性门禁的数据不会参与最快排名。</p>
      <div class="table-scroll"><table><thead><tr><th>Batch</th><th>Torch</th><th>Ascend C</th><th>PyPTO</th><th>可信最快</th></tr></thead><tbody>${comparisonRows || '<tr><td colspan="5">无可用性能记录</td></tr>'}</tbody></table></div>
    </div>
    <div class="section"><h3>详细性能记录</h3><div class="table-scroll"><table class="metric-table"><thead><tr><th>Batch</th><th>路线</th><th>主计算核 µs</th><th>全设备 µs/调用</th><th>AICPU µs/调用</th><th>TFLOPS</th><th>Kernel 类型</th><th>Kernels/调用</th><th>blockDim</th><th>质量</th></tr></thead><tbody>${detailRows || '<tr><td colspan="10">无记录</td></tr>'}</tbody></table></div></div>`;
}

function renderEvidence(operator) {
  const unique = new Map();
  Object.values(operator.performance.batches).forEach(records => ROUTES.forEach(route => {
    const record = records[route];
    if (record) unique.set(`${route}:${record.source}`, { route, ...record });
  }));
  const evidence = [...unique.values()].map(record => `
    <div class="evidence-item"><strong>${ROUTE_NAMES[record.route]} · ${esc(record.integrity)} · ${esc(record.method)}</strong>
      <p>${esc(record.source)}</p><p>SHA256: ${esc(record.sha256)}</p><p>Kernel: ${esc(record.kernel_name)}</p></div>`).join('');
  const source = state.data.source;
  $('#tab-evidence').innerHTML = `<div class="section"><h3>发布源</h3><div class="evidence-item"><strong>${esc(source.release_file)}</strong><p>SHA256: ${esc(source.release_sha256)}</p><p>优先级：${source.precedence.map(esc).join(' → ')}</p><p>performance_matrix_used: ${source.performance_matrix_used}</p></div></div>
    <div class="section"><h3>性能证据</h3><div class="evidence-list">${evidence || '<div class="evidence-item"><p>无可用性能证据。</p></div>'}</div></div>`;
}

function renderLimitations(operator) {
  const limitations = operator.limitations.map(item => `
    <div class="limitation"><strong>${esc(item.severity || 'INFO')} · ${esc(item.route || 'all')}</strong><p>${esc(item.description)}</p></div>`).join('');
  const notes = Object.entries(operator.release_notes).filter(([, value]) => value).map(([key, value]) => `
    <div class="evidence-item"><strong>${esc(key)}</strong><p>${esc(value)}</p></div>`).join('');
  $('#tab-limitations').innerHTML = `<div class="section"><h3>已知限制</h3><div class="limitation-list">${limitations || '<div class="evidence-item"><p>发布源未记录算子级限制。</p></div>'}</div></div>
    <div class="section"><h3>本次更新</h3><div class="evidence-list">${notes || '<div class="evidence-item"><p>无算子级更新记录。</p></div>'}</div></div>`;
}

function showDetail(name) {
  const operator = state.data.operators[name];
  if (!operator) return;
  state.selected = name;
  $('#detail-name').textContent = name;
  $('#detail-formula').textContent = operator.formula;
  $('#detail-status').textContent = operator.status_zh;
  $('#detail-status').className = `status ${statusClass(operator.status)}`;
  $('#detail-warnings').innerHTML = operator.warnings.map(warning => `<div class="warning">⚠ ${esc(warning)}</div>`).join('');
  renderOverview(operator); renderCorrectness(operator); renderPerformance(operator); renderEvidence(operator); renderLimitations(operator);
  $('#detail').hidden = false;
  switchTab('overview');
  $('#detail').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function switchTab(name) {
  $$('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.tab === name));
  $$('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === `tab-${name}`));
}

function setupEvents() {
  $('#search').addEventListener('input', event => { state.query = event.target.value; renderOperators(); });
  $('#status-filter').addEventListener('change', event => { state.status = event.target.value; renderOperators(); });
  $('#refresh').addEventListener('click', () => location.reload());
  $('#close-detail').addEventListener('click', () => { $('#detail').hidden = true; state.selected = null; window.scrollTo({ top: 0, behavior: 'smooth' }); });
  $$('.tab').forEach(tab => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));
  $('#toggle-updates').addEventListener('click', event => {
    const updates = $('#updates');
    updates.hidden = !updates.hidden;
    event.target.textContent = updates.hidden ? '展开' : '收起';
  });
}

async function init() {
  try {
    state.data = await loadDashboard();
    renderHeader(state.data); renderSummary(state.data); renderUpdates(state.data); renderOperators(); setupEvents();
    $('#loading').hidden = true; $('#app').hidden = false;
  } catch (error) {
    $('#loading').textContent = `Dashboard 加载失败：${error.message}`;
  }
}

document.addEventListener('DOMContentLoaded', init);
