const goal = document.querySelector('#goal');
const method = document.querySelector('#method');
const dataset = document.querySelector('#dataset');
const models = document.querySelector('#models');
const note = document.querySelector('#selection-note');
const view = document.querySelector('#view');
const runJob = document.querySelector('#run-job');
const stopJob = document.querySelector('#stop-job');
const jobStatus = document.querySelector('#job-status');
const jobResults = document.querySelector('#job-results');
const jobReport = document.querySelector('#job-report');
const jobProgress = document.querySelector('#job-progress');
const jobReportSection = document.querySelector('#job-report-section');
const report = document.querySelector('#report');
let revision = 0;
let validatedMode = null;
let currentJob = null;
let jobActive = false;
let jobStatusAvailable = false;
let timer = null;

function element(tag, text, className) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = text;
  return el;
}
document.querySelector('#two-stage-view').addEventListener('click', async () => {
  const status = document.querySelector('#example-status');
  try {
    const response = await fetch('/api/two-stage-report');
    if (!response.ok) throw new Error('本地合成图文记录无法核验。');
    const data = await response.json();
    document.querySelector('#two-stage-title').textContent = data.title;
    document.querySelector('#two-stage-scope').textContent = data.scope;
    const counts = data.status;
    document.querySelector('#two-stage-counts').textContent = `图文阶段 ${counts.image_completed}/${counts.image_planned} 项；文字阶段 ${counts.text_completed}/${counts.text_planned} 项；失败 ${counts.failed}，中断 ${counts.interrupted}，未执行 ${counts.not_run}，截断 ${counts.truncated}。`;
    document.querySelector('#two-stage-limits').replaceChildren(...data.limitations.map(text => element('p', text)));
    document.querySelector('#two-stage-records').replaceChildren(...data.records.map(row => {
      const block = element('div', '', 'row');
      block.append(element('strong', `${row.stage} · ${row.label}`), element('p', `实际问题：${row.question}`), element('p', `原答：${row.answer}`), element('p', `状态：已完成 · 来源：${row.source}`));
      return block;
    }));
    const section = document.querySelector('#two-stage-report');
    section.hidden = false;
    section.scrollIntoView({behavior:'smooth'});
    status.textContent = '';
  } catch (error) {
    document.querySelector('#two-stage-report').hidden = true;
    status.textContent = ' ' + error.message;
  }
});
const browseDataset = document.querySelector('#browse-dataset');
const browseCategory = document.querySelector('#browse-category');
const gallery = document.querySelector('#dataset-gallery');
const datasetStatus = document.querySelector('#dataset-status');
const datasetPrev = document.querySelector('#dataset-prev');
const datasetNext = document.querySelector('#dataset-next');
let datasetsById = {};
let datasetPage = 1;
let datasetPages = 1;
let browseRevision = 0;
async function loadDatasetPage() {
  const selected = browseDataset.value;
  if (!selected) return;
  const current = ++browseRevision;
  datasetStatus.textContent = '正在读取开发图片…';
  datasetPrev.disabled = datasetNext.disabled = true;
  try {
    const params = new URLSearchParams({category:browseCategory.value, page:String(datasetPage)});
    const response = await fetch(`/api/datasets/${encodeURIComponent(selected)}?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '数据集不可用');
    if (current !== browseRevision) return;
    gallery.replaceChildren();
    for (const item of data.items) {
      const card = element('figure', '', 'dataset-card');
      const preview = document.createElement('img');
      preview.src = item.image_url;
      preview.alt = `SAR 开发图，参考类别${item.label_name}，样本 ${item.sample_id}`;
      preview.loading = 'lazy';
      preview.addEventListener('error', () => { preview.alt = '图片不可用或校验失败'; });
      const caption = element('figcaption', '', 'dataset-caption');
      caption.append(element('strong', item.label_name), element('span', item.sample_id));
      card.append(preview, caption); gallery.append(card);
    }
    datasetPages = data.pages;
    document.querySelector('#dataset-page').textContent = `第 ${data.page} / ${data.pages} 页`;
    datasetStatus.textContent = `当前类别共 ${data.filtered_total} 张，本页 ${data.items.length} 张。`;
    datasetPrev.disabled = data.page <= 1;
    datasetNext.disabled = data.page >= data.pages;
  } catch (error) {
    if (current !== browseRevision) return;
    gallery.replaceChildren();
    document.querySelector('#dataset-page').textContent = '';
    datasetStatus.textContent = '无法读取图片：' + error.message;
  }
}
function showDatasetInfo(data) {
  document.querySelector('#dataset-info').hidden = false;
  document.querySelector('#dataset-source').textContent = data.source;
  document.querySelector('#dataset-stage').textContent = data.stage;
  document.querySelector('#dataset-total').textContent = `${data.total} 张`;
  document.querySelector('#dataset-description').textContent = data.description;
  const limit = document.querySelector('#dataset-limit'); limit.textContent = data.limit; limit.hidden = false;
  const counts = document.querySelector('#dataset-counts'); counts.replaceChildren();
  browseCategory.replaceChildren(new Option('全部类别', ''));
  for (const item of data.classes) {
    counts.append(element('span', `${item.name} ${item.count}`, 'dataset-count'));
    browseCategory.add(new Option(`${item.name}（${item.count}）`, item.id));
  }
  browseCategory.disabled = false;
}
browseDataset.addEventListener('change', () => {
  if (!datasetsById[browseDataset.value]) return;
  datasetPage = 1; showDatasetInfo(datasetsById[browseDataset.value]); loadDatasetPage();
});
browseCategory.addEventListener('change', () => { datasetPage = 1; loadDatasetPage(); });
datasetPrev.addEventListener('click', () => { if (datasetPage > 1) { datasetPage--; loadDatasetPage(); } });
datasetNext.addEventListener('click', () => { if (datasetPage < datasetPages) { datasetPage++; loadDatasetPage(); } });
fetch('/api/datasets').then(async response => {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '数据集资料不可用');
  datasetsById = Object.fromEntries(data.datasets.map(item => [item.id, item]));
  browseDataset.replaceChildren();
  for (const item of data.datasets) browseDataset.add(new Option(item.name, item.id));
  if (!data.datasets.length) throw new Error('没有可展示的数据集');
  browseDataset.disabled = false;
  browseDataset.value = data.datasets[0].id;
  showDatasetInfo(data.datasets[0]);
  await loadDatasetPage();
}).catch(error => { datasetStatus.textContent = '无法读取数据集：' + error.message; });

function reset(select, placeholder) {
  select.replaceChildren(new Option(placeholder, ''));
  select.disabled = true;
}
function selection() { return {goal:goal.value, method:method.value, dataset:dataset.value, models:models.value}; }
function updateActions() {
  runJob.hidden = validatedMode !== 'local_reeval';
  view.hidden = validatedMode !== 'report_only';
  runJob.disabled = validatedMode !== 'local_reeval' || jobActive || !jobStatusAvailable;
  view.disabled = validatedMode !== 'report_only';
}
function clearAfter(stage) {
  if (stage <= 1) reset(method, '请先选择研究目标');
  if (stage <= 2) reset(dataset, '请先选择评价方式');
  if (stage <= 3) reset(models, '请先选择数据范围');
  validatedMode = null;
  updateActions();
  report.hidden = true;
}
async function request(path, params) {
  const response = await fetch(path + '?' + new URLSearchParams(params));
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '后端请求失败');
  return result;
}
async function populate(stage, select, placeholder) {
  const current = ++revision;
  const result = await request('/api/options', {stage, ...selection()});
  if (current !== revision) return;
  reset(select, placeholder);
  for (const item of result.options) {
    select.add(new Option(item.name + (item.status === 'planned' ? ' · 待接入' : ''), item.id));
    select.disabled = false;
  }
  if (!result.options.length) note.textContent = '这一方向尚未接入可运行实验，暂无结果可展示。';
}
function fail(error) { note.textContent = error.message; }
populate('goals', goal, '请选择研究目标').catch(fail);
goal.addEventListener('change', async () => {
  ++revision; clearAfter(1);
  if (!goal.value) { note.textContent = '选择研究目标后，后续选项会按支持情况开放。'; return; }
  note.textContent = '请选择评价方式。';
  try { await populate('methods', method, '请选择评价方式'); } catch (error) { fail(error); }
});
method.addEventListener('change', async () => {
  ++revision; clearAfter(2);
  if (method.value) try { await populate('datasets', dataset, '请选择数据范围'); } catch (error) { fail(error); }
});
dataset.addEventListener('change', async () => {
  ++revision; clearAfter(3);
  if (dataset.value) try { await populate('models', models, '请选择模型'); } catch (error) { fail(error); }
});
models.addEventListener('change', async () => {
  const current = ++revision;
  validatedMode = null; updateActions(); report.hidden = true;
  if (!models.value) return;
  try {
    const validation = await request('/api/validate', selection());
    if (current !== revision) return;
    validatedMode = validation.mode;
    note.textContent = validation.mode === 'local_reeval'
      ? '此测试使用本地 InternVL 2B 和一张已使用开发图，复用已有单步变化图及随机对照，不生成新攻击。运行前请保存工作并关闭其他 GPU 密集程序；单项可能耗时数分钟。'
      : validation.message;
    updateActions();
  } catch (error) { if (current === revision) fail(error); }
});

async function jobRequest(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '任务请求失败');
  return result;
}
const conditionNames = {clean:'原图', surrogate_step:'已有单步变化图', random_control:'随机对照'};
const judgmentNames = {correct:'与参考一致', wrong:'与参考不一致', unresolved:'无法判断'};
function showJob(job) {
  if (!job) { jobStatusAvailable = true; updateActions(); return; }
  jobStatusAvailable = true;
  currentJob = job.id;
  jobActive = ['queued', 'running'].includes(job.status);
  updateActions();
  if (validatedMode === 'local_reeval' && jobActive) note.textContent = '任务已启动，可在下方查看进度。';
  stopJob.disabled = !jobActive;
  jobProgress.hidden = false;
  const states = {queued:'排队中', running:'运行中', completed:'已完成', failed:'失败', interrupted:'已中断'};
  const phases = {imports:'导入依赖', inventory:'核对模型文件', model_loading:'加载模型', inference:'模型推理', generation_done:'保存结果'};
  const elapsed = job.created ? Math.max(0, Math.floor(Date.now() / 1000 - job.created)) : 0;
  const phase = job.phase && phases[job.phase] ? ` · ${phases[job.phase]}（${job.phase_elapsed_seconds} 秒）` : '';
  const sampleNumber = Number.isInteger(job.active_sample_index) ? `第 ${job.active_sample_index + 1} 张的` : '';
  const activeName = job.active_condition ? `正在复评${sampleNumber}${conditionNames[job.active_condition] || '当前条件'}` : (job.message || '');
  jobStatus.textContent = `${states[job.status] || job.status}：${activeName}${phase}${jobActive ? ` · 总耗时 ${elapsed} 秒` : ''}${job.error ? ' ' + job.error : ''}`;
  jobResults.replaceChildren();
  const partial = document.querySelector('#job-partial');
  const partialResults = document.querySelector('#job-partial-results');
  partialResults.replaceChildren();
  partial.hidden = job.status === 'completed' || !(job.results || []).length;
  for (const item of job.results || []) {
    const status = item.status === 'completed' ? `原答：${item.answer}；判读：${judgmentNames[item.judgment] || '未判定'}${item.truncated ? '；达到输出上限' : ''}` : `未完成：${item.status}`;
    const sample = Number.isInteger(item.sample_index) ? `样本 ${item.sample_index + 1} · ` : '';
    const text = `${sample}${conditionNames[item.condition] || item.condition}：${status}`;
    jobResults.append(element('div', text, 'row'));
    partialResults.append(element('div', text, 'row'));
  }
  jobReportSection.hidden = job.status !== 'completed' || !job.report;
  if (!jobReportSection.hidden) {
    const count = Array.isArray(job.sample_indices) ? job.sample_indices.length : 1;
    document.querySelector('#job-report-badge').textContent = `开发实验 · ${count} 张图`;
    document.querySelector('#job-report-scope').textContent = `本地 InternVL 2B · ${count} 张已使用开发图 · 复用已有变化图，不生成新攻击，不是闭源迁移测试。`;
    const descriptions = job.counts
      ? Object.entries(job.counts).map(([condition, c]) => `${conditionNames[condition] || condition}：与参考一致 ${c.correct}、不一致 ${c.wrong}、无法判断 ${c.unresolved} / ${count}`)
      : (job.results || []).map(item => `${conditionNames[item.condition] || item.condition}${judgmentNames[item.judgment] || '未判定'}`);
    const paired = job.paired_attack_random;
    const comparison = paired ? `变化图与随机对照同图比较：两者均错 ${paired.both_wrong}、仅变化图错 ${paired.attack_only_wrong}、仅随机图错 ${paired.random_only_wrong}、两者均正确 ${paired.both_correct}、至少一项无法判断 ${paired.unresolved}。` : '';
    jobReport.textContent = `本次使用本地 InternVL 2B，分别检查原图、已有单步变化图和随机对照。${descriptions.join('；')}。${comparison}本次复用先前生成的图片，没有新生成攻击；自动判读仅适用于简短单类别回答，其他情况需要人工复核。`;
  }
  if (timer) clearTimeout(timer);
  if (jobActive) timer = setTimeout(pollJob, 2000);
}
async function pollJob() {
  try { if (currentJob) showJob((await jobRequest('/api/jobs/' + currentJob)).job); }
  catch (error) { jobStatusAvailable = false; updateActions(); stopJob.disabled = true; jobStatus.textContent = '状态读取失败：' + error.message; }
}
runJob.addEventListener('click', async () => {
  runJob.disabled = true;
  let submissionStarted = false;
  jobProgress.hidden = false;
  jobStatus.textContent = '正在提交任务，请稍候…';
  note.textContent = '正在提交测试任务，请勿重复点击。';
  jobProgress.scrollIntoView({behavior:'smooth'});
  try {
    const chosen = selection();
    const check = await request('/api/validate', chosen);
    if (check.mode !== 'local_reeval') throw new Error('当前配置不能启动本地复评。');
    submissionStarted = true;
    const result = await jobRequest('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({kind:'local_reeval', selection:chosen})});
    currentJob = result.id; jobReportSection.hidden = true;
    await pollJob();
    jobProgress.scrollIntoView({behavior:'smooth'});
  } catch (error) {
    const message = submissionStarted
      ? '任务提交状态无法确认，请刷新页面检查最新任务后再操作：' + error.message
      : '任务未启动：' + error.message;
    if (submissionStarted) jobStatusAvailable = false;
    jobStatus.textContent = message;
    note.textContent = message;
    updateActions();
  }
});
stopJob.addEventListener('click', async () => {
  stopJob.disabled = true;
  try { await jobRequest('/api/jobs/' + currentJob + '/cancel', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'}); jobStatus.textContent = '正在停止当前调用…'; await pollJob(); }
  catch (error) { jobStatus.textContent = error.message; await pollJob(); }
});
jobRequest('/api/jobs/latest').then(({job}) => showJob(job)).catch(error => { jobStatusAvailable = false; updateActions(); note.textContent = '无法读取任务状态：' + error.message; });

const labels = {clean:'干净图像', surrogate_step:'单步变化', random_control:'随机对照'};
const exampleStatus = document.querySelector('#example-status');
function reportMetricList(metrics) {
  const list = element('ul', '');
  for (const metric of metrics) {
    const ratio = metric.denominator ? `${metric.numerator}/${metric.denominator}` : '不可计算';
    const row = element('li', `${metric.label} ${ratio}（证据 ${metric.evidence_ids.join('、')}）`);
    if (metric.evidence_ids.length) {
      const link = element('button', '查看原答', 'inline-action');
      link.type = 'button';
      link.addEventListener('click', () => {
        document.querySelector('#example-evidence-details').open = true;
        const original = document.getElementById('report-evidence-' + metric.evidence_ids[0]);
        if (original) {
          original.focus({preventScroll:true});
          original.scrollIntoView({block:'center'});
        }
      });
      row.append(link);
    }
    list.append(row);
  }
  return list;
}

function renderSavedReport(saved, notice, limitations) {
  const container = document.querySelector('#saved-report');
  const status = document.querySelector('#saved-report-notice');
  container.replaceChildren();
  container.hidden = !saved;
  status.hidden = !notice;
  status.textContent = notice || '';
  document.querySelector('#example-template').open = !saved;
  if (!saved) return;
  container.append(element('h3', 'DeepSeek 生成的实验结果分析'),
                   element('p', `${saved.provider} · ${saved.model} · 待人工复核`, 'subtle'));
  for (const section of saved.sections) {
    container.append(element('h4', section.title), element('p', section.analysis),
                     reportMetricList(section.metrics));
  }
  container.append(element('h4', '人工复核补充'),
                   element('p', saved.review_note.analysis), reportMetricList(saved.review_note.metrics));
  container.append(element('h4', '适用范围'), element('p', limitations.join(' ')));
}

const exampleButtons = ['perturbation', 'injection'].map(id => [id, document.querySelector('#example-' + id)]);
for (const [id, button] of exampleButtons) {
  button.addEventListener('click', async () => {
    for (const [, action] of exampleButtons) action.disabled = true;
    exampleStatus.textContent = ' 正在核对历史记录…';
    try {
      const {facts, example, evidence, saved_report, saved_report_notice} = await request('/api/report-inputs/' + id);
      document.querySelector('#example-title').textContent = facts.title;
      document.querySelector('#example-scope').textContent = facts.scope;
      document.querySelector('#injection-risk').hidden = id !== 'injection';
      const analysis = document.querySelector('#example-analysis');
      analysis.replaceChildren();
      for (const item of example) {
        analysis.append(element('h3', item.title), element('p', item.analysis));
        analysis.append(reportMetricList(item.metric_ids.map(key => facts.metrics[key])));
      }
      analysis.append(element('h3', '适用范围'), element('p', facts.limitations.join(' ')));
      renderSavedReport(saved_report, saved_report_notice, facts.limitations);
      const states = facts.statuses;
      document.querySelector('#example-counts').textContent =
        `计划 ${facts.planned} 项，完成 ${states.completed} 项，失败 ${states.failed} 项，中断 ${states.interrupted} 项，未执行 ${states.not_run} 项。`;
      document.querySelector('#example-conditions').replaceChildren(...facts.conditions.map(c => {
        const j = c.judgments;
        const execution = `完成 ${c.completed}/${c.planned}，失败 ${c.failed}，中断 ${c.interrupted}，未执行 ${c.not_run}，截断 ${c.truncated}`;
        const judgments = j ? `；与参考一致 ${j.correct}，不一致 ${j.wrong}，无法判断 ${j.unresolved}` : '';
        return element('p', `${c.model} · ${c.label} · ${execution}${judgments}`, 'row');
      }));
      document.querySelector('#example-evidence').replaceChildren(...evidence.map(row => {
        const result = row.judgment ? ` · ${judgmentNames[row.judgment]}` : '';
        const original = element('p', `${row.id} · ${row.sample} · ${row.model} · ${row.condition_label} · ${row.answer}${result}`, 'row');
        original.id = 'report-evidence-' + row.id;
        original.tabIndex = -1;
        return original;
      }));
      const section = document.querySelector('#report-example');
      section.hidden = false;
      section.scrollIntoView({behavior:'smooth'});
      exampleStatus.textContent = '';
    } catch (error) {
      exampleStatus.textContent = ' ' + error.message;
    } finally {
      for (const [, action] of exampleButtons) action.disabled = false;
    }
  });
}
if (new URLSearchParams(window.location.search).get('report') === 'injection') {
  document.querySelector('#example-injection').click();
}
document.querySelector('#baseline-view').addEventListener('click', async () => {
  try {
    const response = await fetch('/api/clean-report');
    if (!response.ok) throw new Error('81 图开发报告暂不可用，请检查本地记录。');
    const data = await response.json();
    document.querySelector('#baseline-scope').textContent = data.scope;
    const cards = document.querySelector('#baseline-cards'); const rows = document.querySelector('#baseline-rows');
    cards.replaceChildren(); rows.replaceChildren();
    const names = {internvl1b:'InternVL2.5 1B', internvl2b:'InternVL2.5 2B', mplug:'mPLUG-Owl3 1B', glm:'GLM-edge-v 2B'};
    const classes = {ship:'船舶', aircraft:'飞机', car:'汽车', tank:'储油罐', bridge:'桥梁', harbor:'港口'};
    for (const [key, model] of Object.entries(data.models)) {
      const card = element('div', '', 'card'); card.append(element('h3', names[key]));
      const c = model.counts;
      card.append(element('div', `总计：正确 ${c.correct || 0}，错误 ${c.wrong || 0}，无法判断 ${c.unresolved || 0} / ${model.total}`, 'metric'));
      for (const [cls, label] of Object.entries(classes)) {
        const v = model.per_class[cls]; const line = element('div', '', 'metric');
        line.append(element('span', label), element('b', `${v.correct} / ${v.total}`)); card.append(line);
      }
      cards.append(card);
      for (const item of model.items) rows.append(element('div', `${names[key]} · ${item.sample_id} · 参考 ${classes[item.reference]} · ${item.raw_answer} · ${item.judgment}`, 'row'));
    }
    const section = document.querySelector('#baseline'); section.hidden = false; section.scrollIntoView({behavior:'smooth'});
  } catch (error) { note.textContent = error.message; }
});
function renderHistoricalAnalysis(data) {
  const total = data.sample_count;
  const source = data.counts.mplug;
  const target = data.counts.internvl2b;
  const sourcePair = data.paired.mplug;
  const targetStep = target.surrogate_step;
  const sharedErrors = data.items.filter(item => ['mplug', 'internvl2b'].every(model =>
    item.judgments[model].clean.judgment === 'correct' &&
    item.judgments[model].surrogate_step.judgment === 'wrong'
  )).length;
  const sections = [
    ['评估结论',
      `在这 ${total} 张已知两模型曾共同答对的 SAR 开发图中，mPLUG 1B 在单步变化图上的正确数为 ${source.surrogate_step.correct}/${total}，原图和同预算随机对照分别为 ${source.clean.correct}/${total} 和 ${source.random_control.correct}/${total}。使用同一批变化图时，本地 InternVL2B 答对 ${targetStep.correct}/${total}，有 ${targetStep.wrong} 例明确错误和 ${targetStep.unresolved} 例无法判断。`],
    ['mPLUG 1B 的潜在问题',
      `逐图对照发现 ${sourcePair.attack_only_wrong} 例仅在单步变化下明确答错，另有 ${source.surrogate_step.unresolved} 例变化图回答无法判定。这提示模型的分类结论可能对特定像素变化敏感。若下游系统直接使用单一类别，错误类别可能进入后续判断，多类别答复也需要能被保留为不确定结果。`],
    ['InternVL2B 与迁移证据',
      `本地 InternVL2B 对同一批变化图的明确错误为 ${targetStep.wrong}/${total}，随机对照的明确错误为 ${target.random_control.wrong}/${total}。两个模型在同一张图上均由原图答对变为变化图答错的情况有 ${sharedErrors}/${total} 例。这里的变化针对 mPLUG 构造，InternVL2B 自身面对针对它构造的变化时是否稳定，仍需单独评估。`],
    ['适用范围',
      '样本按历史共同答对筛选，每类仅两张，每图只有一个随机对照。参考标签尚待领域复核，原答采用既有单人判读。结论限于这组显示图像上的扰动开发观察，无法估计总体错误率或稳定迁移能力。']
  ];
  const analysis = document.querySelector('#historical-analysis');
  analysis.replaceChildren();
  for (const [heading, paragraph] of sections) {
    analysis.append(element('h4', heading), element('p', paragraph));
    if (heading !== 'mPLUG 1B 的潜在问题') continue;
    const examples = element('ul', '');
    for (const item of data.items) {
      const answer = item.judgments.mplug.surrogate_step;
      if (answer.judgment === 'correct') continue;
      examples.append(element('li', `${item.id}（参考类别 ${item.reference}）回答「${answer.answer}」，既有判读为${judgmentNames[answer.judgment]}。`));
    }
    if (examples.children.length) analysis.append(examples);
  }
}

view.addEventListener('click', async () => {
  view.disabled = true;
  try {
    const check = await request('/api/validate', selection());
    if (check.mode !== 'report_only') throw new Error('当前配置不是历史报告。');
    const response = await fetch('/api/report');
    if (!response.ok) throw new Error('本地实验记录不可用，请检查运行产物。');
    const data = await response.json();
    document.querySelector('#scope').textContent = data.scope;
    document.querySelector('#description').textContent = `${data.description} ${data.provenance}`;
    renderHistoricalAnalysis(data);
    const cards = document.querySelector('#cards'); cards.replaceChildren();
    for (const [key, name] of [['mplug','源模型 · mPLUG 1B'], ['internvl2b','目标模型 · InternVL2B']]) {
      const card = element('div', '', 'card'); card.append(element('h3', name));
      for (const condition of Object.keys(labels)) {
        const c = data.counts[key][condition]; const row = element('div', '', 'metric');
        row.append(element('span', labels[condition]), element('b', `正确 ${c.correct} · 错误 ${c.wrong} · 无法判断 ${c.unresolved} / ${data.sample_count}`)); card.append(row);
      }
      const p = data.paired[key];
      card.append(element('p', `变化图与随机对照同图配对 / ${data.sample_count}：均错 ${p.both_wrong}，仅变化图错 ${p.attack_only_wrong}，仅随机对照错 ${p.random_only_wrong}，均正确 ${p.both_correct}，至少一项无法判断 ${p.unresolved}。`, 'subtle'));
      cards.append(card);
    }
    const rows = document.querySelector('#rows'); rows.replaceChildren();
    for (const item of data.items) {
      const row = element('div', '', 'row'); row.append(element('strong', `${item.id} · 参考类别 ${item.reference}`));
      for (const [key, name] of [['mplug','mPLUG'], ['internvl2b','InternVL2B']]) row.append(element('div', `${name}：${Object.keys(labels).map(c => `${labels[c]}「${item.judgments[key][c].answer}」（${judgmentNames[item.judgments[key][c].judgment] || '未知判读'}${item.judgments[key][c].hit_limit ? '，达到长度上限' : ''}）`).join('；')}`));
      rows.append(row);
    }
    report.hidden = false; report.scrollIntoView({behavior:'smooth'});
  } catch (error) { note.textContent = error.message; }
  finally { updateActions(); }
});

const latestResultsBtn = document.querySelector('#latest-results-view');
if (latestResultsBtn) {
  latestResultsBtn.addEventListener('click', async () => {
    latestResultsBtn.disabled = true;
    exampleStatus.textContent = ' 正在核验最新实验记录…';
    try {
      const response = await fetch('/api/latest-results');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '最新实验记录无法核验');
      const cards = document.querySelector('#latest-results-cards');
      const limits = document.querySelector('#latest-results-limitations');
      cards.replaceChildren(); limits.replaceChildren();
      const makeCard = (title, description) => {
        const card = element('div', '', 'card');
        card.append(element('h3', title), element('p', description, 'subtle'));
        cards.append(card); return card;
      };
      const addMetric = (card, label, value) => {
        const line = element('div', '', 'metric');
        line.append(element('span', label), element('b', value)); card.append(line);
      };
      const local = makeCard('本地视觉原型迁移 · InternVL2B', '17 张受限工程配对图，16/255；目标未参与本组梯度生成。');
      for (const x of data.local_metrics) addMetric(local, x.label, `错 ${x.wrong}/${x.denominator} · 预设命中 ${x.target_hit}/${x.denominator}`);
      const engineering = makeCard('工程快照四条件 · 8/255', '同图配对 17 张；源模型 mPLUG，目标模型 InternVL2B。');
      for (const x of data.engineering8) addMetric(engineering, x.label, `源正确 ${x.source_correct}/${x.denominator} · 源命中 ${x.source_hit}/${x.denominator} · 目标正确 ${x.target_correct}/${x.denominator} · 目标命中 ${x.target_hit}/${x.denominator}`);
      for (const group of data.cloud_clean) {
        const card = makeCard(group.title, '17 张工程快照原图；先列完整原图分母。');
        for (const x of group.metrics) addMetric(card, x.label, `与参考一致 ${x.correct}/${x.denominator} · 截断 ${x.truncated}/${x.denominator}`);
      }
      for (const group of data.cloud_groups) {
        const card = makeCard(group.title, `已完成 ${group.completed}/${group.planned} 次；各条件分母是对应提示下原图答对的子集。`);
        for (const x of group.metrics) addMetric(card, x.label, x.completed ? `错 ${x.wrong}/${x.completed} · 命中 ${x.target_hit}/${x.completed} · 完成 ${x.completed}/${x.planned}` : `暂无完成结果 · 完成 0/${x.planned}`);
      }
      const prompt = makeCard('百炼提示开发对照', `新增 14 图原图与暂定参考一致 ${data.clean14.agree}/${data.clean14.planned}；以下为同一批开发图的提示比较。`);
      for (const x of data.prompt_metrics) addMetric(prompt, x.label, `与暂定参考一致 ${x.agree}/${x.denominator}`);
      for (const pilot of data.pilots) {
        const card = makeCard(pilot.title, '非专家暂定参考；受限工程探索。');
        for (const x of pilot.metrics) addMetric(card, x.label, `不一致 ${x.wrong}/${x.denominator} · 预设命中 ${x.target_hit}/${x.denominator}`);
      }
      limits.append(element('strong', '结果范围与限制'));
      for (const text of data.limitations) limits.append(element('p', text));
      const section = document.querySelector('#latest-results');
      section.hidden = false; section.scrollIntoView({behavior:'smooth'});
      exampleStatus.textContent = '';
    } catch (error) {
      document.querySelector('#latest-results').hidden = true;
      exampleStatus.textContent = ' ' + error.message;
    }
    finally { latestResultsBtn.disabled = false; }
  });
}

const recentSarBtn = document.querySelector('#recent-sar-view');
if (recentSarBtn) {
  recentSarBtn.addEventListener('click', async () => {
    recentSarBtn.disabled = true;
    exampleStatus.textContent = ' 正在核验近期 SAR 实验哈希…';
    try {
      const response = await fetch('/api/recent-experiments');
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '近期 SAR 实验记录无法核验');
      const {
        tisimi_six: tis,
        hardlabel_six: hl,
        qwen2vl_sar: qw,
        tisimi_ablation_six: abl,
        engineering_cohort_v1: cohort,
      } = data.suites;
      const cards = document.querySelector('#recent-sar-cards');
      const limits = document.querySelector('#recent-sar-limits');
      const rows = document.querySelector('#recent-sar-rows');
      cards.replaceChildren();
      limits.replaceChildren();
      rows.replaceChildren();

      const metricLine = (label, val) => {
        const line = element('div', '', 'metric');
        line.append(element('span', label), element('b', val));
        return line;
      };

      // Card 1: TI-SI-MI 6-image transfer
      const c1 = element('div', '', 'card');
      c1.append(element('h3', tis.title));
      c1.append(
        metricLine('mPLUG 历史 PGD4 (ε=4/255)', `错 ${tis.counts.mplug.pgd4.wrong} · 无法判断 ${tis.counts.mplug.pgd4.unresolved} / ${tis.denominator}`),
        metricLine('mPLUG 新 TI-SI-MI4 (ε=4/255)', `错 ${tis.counts.mplug.tisimi4.wrong}/${tis.denominator}（定向命中 ${tis.counts.mplug.tisimi4.target_hit}/${tis.denominator}）`),
        metricLine('mPLUG 新 TI-SI-MI8 (ε=8/255)', `错 ${tis.counts.mplug.tisimi8.wrong}/${tis.denominator}（定向命中 ${tis.counts.mplug.tisimi8.target_hit}/${tis.denominator}）`),
        metricLine('InternVL2B 跨模型迁移 (ε=8/255)', `错 ${tis.counts.internvl2b.tisimi8.wrong}/${tis.denominator}（定向命中 ${tis.counts.internvl2b.tisimi8.target_hit}/${tis.denominator}）`),
        metricLine('双模型同预算随机对照 (ε=8/255)', `mPLUG 错 ${tis.counts.mplug.random8.wrong}/${tis.denominator} · InternVL2B 错 ${tis.counts.internvl2b.random8.wrong}/${tis.denominator}`),
      );
      cards.append(c1);

      // Card 2: Visual loss ablation + magnitude-matched random sign controls (11/21-22)
      if (abl) {
        const cAbl = element('div', '', 'card');
        cAbl.append(element('h3', abl.title));
        const s = abl.summary;
        cAbl.append(
          metricLine('源模型 mPLUG 目标命中（原方法 / 去视觉损失）', `${s.mplug_full_target_hit}/${abl.denominator} · ${s.mplug_ablation_target_hit}/${abl.denominator}`),
          metricLine('目标模型 InternVL2B 目标命中（原方法 / 去视觉损失）', `${s.internvl2b_full_target_hit}/${abl.denominator} · ${s.internvl2b_ablation_target_hit}/${abl.denominator}`),
          metricLine('InternVL2B 保幅随机符号对照（第 1 / 2 组正确）', `${s.internvl2b_random_sign_1_correct}/${abl.denominator} · ${s.internvl2b_random_sign_2_correct}/${abl.denominator}`),
          metricLine('历史调用失败记录（已保留）', `${s.prior_failures} 次 process_failure`),
        );
        cards.append(cAbl);
      }

      // Card 3: Hard-label 6-image query
      const c2 = element('div', '', 'card');
      c2.append(element('h3', hl.title));
      const hls = hl.summary;
      const linfs = hl.items.map(x => x.linf).filter(v => Number.isFinite(v));
      const linfRange = linfs.length ? `${Math.min(...linfs)}–${Math.max(...linfs)} / 255` : '未知';
      c2.append(
        metricLine('攻击完成与最终明确错', `完成 ${hls.passed}/${hls.planned} · 明确错 ${hls.final_wrong}/${hl.denominator}`),
        metricLine('同掩膜同 L∞ 随机对照错', `${hls.ctrl_wrong} / ${hl.denominator}`),
        metricLine('查询错且随机对照正确（净胜）', `${hls.final_wrong_and_ctrl_correct} / ${hl.denominator}`),
        metricLine('最大像素差范围（非微扰）', linfRange),
      );
      cards.append(c2);

      // Card 4: Qwen2-VL-2B SAR baseline + injection
      const c3 = element('div', '', 'card');
      c3.append(element('h3', qw.title));
      const qb = qw.baseline;
      const qi = qw.injection.summary;
      c3.append(
        metricLine(`6 图基线 · 零样本 (/${qb.denominator})`, `正确 ${qb.zeroshot_counts.correct} · 错 ${qb.zeroshot_counts.wrong} · 无法判断 ${qb.zeroshot_counts.unresolved}`),
        metricLine(`6 图基线 · 长领域提示 (/${qb.denominator})`, `正确 ${qb.domain_prior_counts.correct} · 错 ${qb.domain_prior_counts.wrong} · 无法判断 ${qb.domain_prior_counts.unresolved}`),
        metricLine(`3 图注入 · 空白栏分类正确`, `${qi.blank_correct} / ${qi.denominator}`),
        metricLine(`3 图注入 · 攻击短句独立 OCR 精确`, `${qi.attack_ocr_exact} / ${qi.denominator}`),
        metricLine(`3 图注入 · 攻击 / 可信正向含 REVIEW`, `攻击 ${qi.attack_review}/${qi.denominator} · 正向 ${qi.user_positive_review}/${qi.denominator}`),
      );
      cards.append(c3);

      // Card 5: Limited Engineering Evaluation Cohort v1 (11/23)
      if (cohort) {
        const cCoh = element('div', '', 'card');
        cCoh.append(element('h3', cohort.title));
        const cc = cohort.class_counts;
        cCoh.append(
          metricLine('快照状态与固定容量', `已冻结待评测（${cohort.denominator} 张，模型调用 0）`),
          metricLine('六类分布（船/飞/车/罐/桥/港）', `${cc.ship} / ${cc.aircraft} / ${cc.car} / ${cc.tank} / ${cc.bridge} / ${cc.harbor}`),
          metricLine('配对四条件与容量上限', `${cohort.conditions.join(' / ')}（原图≤${cohort.limits.baseline_images} · 扰动≤${cohort.limits.paired_images}）`),
          metricLine('HRSID 坐标线索移出并单列', `${cohort.excluded_hrsid_count} 张（已由 2 张 ShipDataset 替补）`),
        );
        cards.append(cCoh);
      }

      // Combined limitations
      limits.append(element('strong', '重要局限与解释边界'));
      const allLimits = [
        ...tis.limitations,
        ...(abl ? abl.limitations : []),
        ...hl.limitations,
        ...qw.limitations,
        ...(cohort ? cohort.limitations : []),
      ];
      for (const line of allLimits) {
        limits.append(element('p', line));
      }

      // Per-sample details
      for (const item of tis.items) {
        const m = item.models.mplug;
        const iv = item.models.internvl2b;
        rows.append(element('div',
          `[TI-SI-MI] ${item.sample_id}（参考 ${item.reference} → 目标 ${item.target_category}）| mPLUG: tisimi4「${m.tisimi4.answer}」/ tisimi8「${m.tisimi8.answer}」/ rand8「${m.random8.answer}」| InternVL2B: tisimi4「${iv.tisimi4.answer}」/ tisimi8「${iv.tisimi8.answer}」/ rand8「${iv.random8.answer}」`,
          'row'));
      }
      if (abl) {
        for (const item of abl.items) {
          rows.append(element('div',
            `[视觉消融/随机符号] ${item.sample_id}（${item.reference} → ${item.target}）| mPLUG 原/消融：「${item.models.mplug.full}」/「${item.models.mplug.ablation}」| InternVL2B 原/消融：「${item.models.internvl2b.full}」/「${item.models.internvl2b.ablation}」| 随机符号 1/2：「${item.random_sign[0].answer}」/「${item.random_sign[1].answer}」`,
            'row'));
        }
      }
      for (const item of hl.items) {
        rows.append(element('div',
          `[硬标签查询] ${item.sample_id}（参考 ${item.reference}）| ${item.queries} 次查询 →「${item.final_answer}」（L∞=${item.linf}/255, MAE=${item.mae}/255, 面积 ${(item.changed_ratio * 100).toFixed(1)}%）| 同掩膜随机对照：「${item.ctrl_answer}」（${judgmentNames[item.ctrl_judgment] || item.ctrl_judgment}）`,
          'row'));
      }
      for (const pred of qw.injection.predictions) {
        rows.append(element('div',
          `[Qwen2-VL 注入] ${pred.sample_id}（参考 ${pred.reference}）| 含 REVIEW=${pred.contains_REVIEW ? '是' : '否'} | 原答：「${pred.answer}」`,
          'row'));
      }
      if (cohort) {
        for (const item of cohort.items) {
          rows.append(element('div',
            `[工程快照 v1] ${item.sample_id}（${item.reference} · ${item.source_family} · ${item.image_size[0]}×${item.image_size[1]}）| 跨集同尺寸已查清 ${item.development_pairs.examined_clear_of_flags}/413（未覆盖 ${item.development_pairs.unsupported}）| 准入：${item.admission_basis}`,
            'row'));
        }
      }

      const section = document.querySelector('#recent-sar-report');
      section.hidden = false;
      section.scrollIntoView({behavior:'smooth'});
      exampleStatus.textContent = '';
    } catch (error) {
      exampleStatus.textContent = ' ' + error.message;
    } finally {
      recentSarBtn.disabled = false;
    }
  });
}
