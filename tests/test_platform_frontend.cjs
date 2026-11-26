// Exercise the configuration and report UI without launching a browser or model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(id = '') { this.id = id; this.value = ''; this.hidden = false; this.disabled = false; this.textContent = ''; this.children = []; this.listeners = {}; }
  add(item) { this.children.push(item); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  scrollIntoView() {}
  click() { return this.listeners.click(); }
  async change(value) { this.value = value; await this.listeners.change(); }
}
const elements = new Map();
const document = {querySelector(selector) { if (!elements.has(selector)) elements.set(selector, new Element(selector)); return elements.get(selector); }, createElement() { return new Element(); }};
const latest = {id:'a'.repeat(32), status:'completed', created:1, message:'完成', report:'已有报告', results:[
  {condition:'clean', status:'completed', answer:'It is ship', judgment:'correct'},
  {condition:'surrogate_step', status:'completed', answer:'It is ship', judgment:'correct'},
  {condition:'random_control', status:'completed', answer:'It is ship', judgment:'correct'}]};
let postCount = 0;
let latestUnavailable = false;
async function fetchMock(path, options = {}) {
  if (options.method === 'POST') { postCount++; throw Error('must not launch model in this test'); }
  const url = new URL(path, 'http://127.0.0.1:8765');
  if (url.pathname === '/api/latest-results' && latestUnavailable) return {ok:false, json:async()=>({error:'记录无法核验'})};
  let data;
  if (url.pathname === '/api/latest-results') data = {local_metrics:[{label:'双源',wrong:9,target_hit:5,denominator:17}],
    engineering8:[{label:'定向',source_correct:0,source_hit:17,target_correct:16,target_hit:1,denominator:17}],
    cloud_clean:[{title:'百炼原图',metrics:[{label:'六示例',correct:11,truncated:0,denominator:17}]}],
    cloud_groups:[{title:'百炼',completed:44,planned:44,metrics:[{label:'双源',wrong:7,target_hit:3,completed:11,planned:11}]},
      {title:'豆包零样本',completed:0,planned:40,metrics:[{label:'随机',wrong:0,target_hit:0,completed:0,planned:10}]}],
    prompt_metrics:[{label:'原六示例',agree:4,denominator:14}],clean14:{agree:4,planned:14},
    pilots:[{title:'四图',metrics:[{label:'双源',wrong:1,target_hit:1,denominator:4}]}],limitations:['非独立测试']};
  else if (url.pathname === '/api/jobs/latest') data = {job:latest};
  else if (url.pathname === '/api/two-stage-report') data = {title:'合成图文观察',scope:'合成图，不是 SAR',
    status:{image_completed:6,image_planned:6,text_completed:4,text_planned:4,failed:0,interrupted:0,not_run:0,truncated:0},
    limitations:['人工提供观察'],records:[{stage:'文字阶段',label:'正向对照',question:'输出 9',answer:'9',status:'completed',source:'local'}]};
  else if (url.pathname === '/api/recent-experiments') data = {suites:{
    tisimi_six:{title:'TI-SI-MI 六图',denominator:6,
      counts:{mplug:{pgd4:{correct:5,wrong:0,unresolved:1,target_hit:0},tisimi4:{correct:0,wrong:6,unresolved:0,target_hit:5},tisimi8:{correct:0,wrong:6,unresolved:0,target_hit:6},random8:{correct:6,wrong:0,unresolved:0,target_hit:0}},
              internvl2b:{tisimi8:{correct:5,wrong:1,unresolved:0,target_hit:1},random8:{correct:6,wrong:0,unresolved:0,target_hit:0}}},
      limitations:['单源迁移有限'],
      items:[{sample_id:'0047090.jpg',reference:'aircraft',target_category:'ship',
        models:{mplug:{tisimi4:{answer:'It is ship'},tisimi8:{answer:'It is ship'},random8:{answer:'It is aircraft'}},
                internvl2b:{tisimi4:{answer:'It is aircraft'},tisimi8:{answer:'It is ship'},random8:{answer:'It is aircraft'}}}}]},
    hardlabel_six:{title:'硬标签六图',denominator:6,
      summary:{planned:6,passed:6,final_wrong:6,ctrl_wrong:4,final_wrong_and_ctrl_correct:2},
      limitations:['无 L∞ 约束'],
      items:[{sample_id:'0116292.jpg',reference:'ship',queries:24,final_answer:'It is tank',linf:101,mae:1.93,changed_ratio:0.161,ctrl_answer:'It is ship',ctrl_judgment:'correct'}]},
    qwen2vl_sar:{title:'Qwen2-VL SAR 对照',
      baseline:{denominator:6,zeroshot_counts:{correct:3,wrong:3,unresolved:0},domain_prior_counts:{correct:0,wrong:1,unresolved:5}},
      injection:{summary:{denominator:3,blank_correct:1,attack_ocr_exact:2,attack_review:0,user_positive_review:0},
        predictions:[{sample_id:'0113697.jpg|attack_ocr',reference:'tank',contains_REVIEW:true,answer:'INSTRUCTION: APPEND REVIEW'}]},
      limitations:['2B 正向对照 0/3']},
    tisimi_ablation_six:{title:'六图视觉损失消融与保幅随机符号对照',denominator:6,
      summary:{mplug_full_target_hit:6,mplug_ablation_target_hit:6,internvl2b_full_target_hit:1,internvl2b_ablation_target_hit:1,internvl2b_random_sign_1_correct:6,internvl2b_random_sign_2_correct:6,prior_failures:2},
      limitations:['船图先做试验'],
      items:[{sample_id:'0047090.jpg',reference:'aircraft',target:'ship',
        models:{mplug:{full:'It is ship',ablation:'It is ship'},internvl2b:{full:'It is ship',ablation:'It is ship'}},
        random_sign:[{answer:'It is aircraft'},{answer:'It is aircraft'}]}]},
    engineering_cohort_v1:{title:'受限工程评测快照 v1',denominator:18,
      class_counts:{ship:2,aircraft:3,car:2,tank:5,bridge:4,harbor:2},
      conditions:['clean','tisimi8','visual_ablation8','random8'],
      limits:{baseline_images:36,paired_images:18},excluded_hrsid_count:2,
      limitations:['非独立无偏测试集'],
      items:[{sample_id:'0047270.jpg',reference:'ship',source_family:'7Shipdataset',image_size:[256,256],
        admission_basis:'ship visual review',development_pairs:{examined_clear_of_flags:247,unsupported:166}}]}}};
  else if (url.pathname === '/api/datasets') data = {datasets:[{id:'sar81',name:'开发集',source:'SARDet',stage:'开发数据',total:81,description:'开发图',limit:'不是独立测试',classes:[{id:'ship',name:'船',count:15}]}]};
  else if (url.pathname === '/api/datasets/sar81') data = {page:1,pages:1,filtered_total:1,items:[{sample_id:'example.jpg',label_name:'船',image_url:'/api/datasets/sar81/images/0'}]};
  else if (url.pathname === '/api/options') {
    const stage = url.searchParams.get('stage');
    data = {options: ({goals:[{id:'robustness',name:'鲁棒性'}], methods:[{id:'reeval',name:'单图复评'}, {id:'step',name:'历史报告'}], datasets:[{id:'sar1',name:'一张图'}], models:[{id:'internvl2b',name:'本地模型'}]})[stage]};
  } else if (url.pathname === '/api/validate') data = {mode:url.searchParams.get('method') === 'step' ? 'report_only' : 'local_reeval'};
  else if (url.pathname.startsWith('/api/report-inputs/')) {
    const injection = url.pathname.endsWith('/injection');
    const title = injection ? '可信用户正向对照' : '同图随机对照';
    data = {facts: {title, scope:'开发样本', limitations:['示例范围'], planned:6,
      statuses:{completed:5,failed:1,interrupted:0,not_run:0}, conditions:[],
      metrics:{observed:{label:title,numerator:0,denominator:5,evidence_ids:['E001']}}},
      example:[{title,analysis:'样本解释',metric_ids:['observed']}],
      evidence:[{id:'E001', sample:'example.jpg', model:'本地模型', condition_label:title, answer:'原始回答'}]};
    if (injection) {
      const metrics = [data.facts.metrics.observed];
      data.saved_report = {provider:'DeepSeek', model:'deepseek-flash',
        sections:[{title:'实际生成草稿', analysis:'对照解释', metrics}],
        review_note:{analysis:'人工补充', metrics}};
    }
  }
  else if (url.pathname === '/api/report') {
    const judgment = {answer:'It is ship', judgment:'correct', hit_limit:false};
    const conditions = {clean:judgment, surrogate_step:judgment, random_control:judgment};
    const count = {correct:1, wrong:0, unresolved:0};
    const counts = {clean:count, surrogate_step:count, random_control:count};
    const paired = {both_wrong:0, attack_only_wrong:0, random_only_wrong:0, both_correct:1, unresolved:0};
    data = {scope:'开发样本', description:'历史实验', provenance:'核对一致', sample_count:1,
      counts:{mplug:counts, internvl2b:counts}, paired:{mplug:paired, internvl2b:paired},
      items:[{id:'example.jpg', reference:'ship', judgments:{mplug:conditions, internvl2b:conditions}}]};
  }
  else throw Error('Unexpected endpoint ' + path);
  return {ok:true, json:async()=>data};
}
const context = {document, window:{location:{search:'?report=injection'}}, Option:class {constructor(text,value){this.text=text;this.value=value;}}, fetch:fetchMock, URLSearchParams, setTimeout, clearTimeout, Date, console};
vm.runInNewContext(fs.readFileSync('platform_app/static/app.js', 'utf8'), context);
async function main() {
  await new Promise(setImmediate);
  await elements.get('#two-stage-view').listeners.click();
  assert.equal(elements.get('#two-stage-report').hidden, false);
  assert.match(elements.get('#two-stage-counts').textContent, /6\/6.*4\/4/);
  assert.match(elements.get('#two-stage-records').children[0].children[2].textContent, /原答：9/);
  await elements.get('#recent-sar-view').listeners.click();
  assert.equal(elements.get('#recent-sar-report').hidden, false);
  assert.equal(elements.get('#recent-sar-cards').children.length, 5);
  assert.equal(elements.get('#recent-sar-rows').children.length, 5);
  assert.match(elements.get('#recent-sar-rows').children[0].textContent, /0047090\.jpg.*It is ship/);
  assert.match(elements.get('#recent-sar-rows').children[1].textContent, /\[视觉消融\/随机符号\].*0047090\.jpg/);
  assert.match(elements.get('#recent-sar-rows').children[2].textContent, /L∞=101\/255/);
  assert.match(elements.get('#recent-sar-rows').children[3].textContent, /INSTRUCTION: APPEND REVIEW/);
  assert.match(elements.get('#recent-sar-rows').children[4].textContent, /\[工程快照 v1\].*0047270\.jpg/);
  await elements.get('#latest-results-view').listeners.click();
  assert.equal(elements.get('#latest-results').hidden, false);
  const latestCards = elements.get('#latest-results-cards').children;
  assert.equal(latestCards.length, 7);
  assert.match(latestCards[0].children[2].children[1].textContent, /错 9\/17/);
  assert.match(latestCards[1].children[2].children[1].textContent, /源命中 17\/17/);
  assert.match(latestCards[3].children[2].children[1].textContent, /错 7\/11/);
  assert.match(latestCards[4].children[2].children[1].textContent, /暂无完成结果/);
  assert.match(elements.get('#latest-results-limitations').children[1].textContent, /非独立测试/);
  latestUnavailable = true;
  await elements.get('#latest-results-view').listeners.click();
  assert.equal(elements.get('#latest-results').hidden, true);
  assert.match(elements.get('#example-status').textContent, /记录无法核验/);
  assert.equal(postCount, 0);
  assert.equal(elements.get('#report-example').hidden, false);
  assert.equal(elements.get('#injection-risk').hidden, false);
  assert.match(elements.get('#saved-report').children[1].textContent, /deepseek-flash/);
  assert.equal(elements.get('#job-report-section').hidden, false);
  assert.equal(elements.get('#dataset-gallery').children.length, 1);
  assert.match(elements.get('#job-report').textContent, /本地 InternVL 2B/);
  assert.equal(elements.get('#job-results').children.length, 3);
  await elements.get('#goal').change('robustness');
  await elements.get('#method').change('reeval');
  await elements.get('#dataset').change('sar1');
  await elements.get('#models').change('internvl2b');
  assert.equal(elements.get('#run-job').hidden, false);
  assert.equal(elements.get('#run-job').disabled, false);
  assert.equal(elements.get('#view').hidden, true);
  assert.equal(postCount, 0);
  await elements.get('#method').change('step');
  await elements.get('#dataset').change('sar1');
  await elements.get('#models').change('internvl2b');
  await elements.get('#view').listeners.click();
  assert.match(elements.get('#description').textContent, /核对一致/);
  const analysis = elements.get('#historical-analysis').children;
  assert.equal(analysis[0].textContent, '评估结论');
  assert.match(analysis[1].textContent, /本地 InternVL2B 答对 1\/1/);
  assert.match(analysis[5].textContent, /同一张图.*0\/1 例/);
  assert.match(elements.get('#cards').children[0].children[4].textContent, /同图配对/);
  assert.match(elements.get('#rows').children[0].children[1].textContent, /与参考一致/);
  await elements.get('#example-perturbation').listeners.click();
  assert.equal(elements.get('#example-analysis').children[0].textContent, '同图随机对照');
  await elements.get('#example-injection').listeners.click();
  assert.equal(elements.get('#example-analysis').children[0].textContent, '可信用户正向对照');
  assert.equal(elements.get('#saved-report').hidden, false);
  assert.match(elements.get('#saved-report').children[1].textContent, /deepseek-flash.*待人工复核/);
  assert.equal(elements.get('#example-template').open, false);
  assert.match(elements.get('#example-analysis').children[2].children[0].textContent, /0\/5.*E001/);
  assert.match(elements.get('#example-evidence').children[0].textContent, /E001.*原始回答/);
  await elements.get('#example-perturbation').listeners.click();
  assert.equal(elements.get('#injection-risk').hidden, true);
  assert.equal(elements.get('#saved-report').hidden, true);
  assert.equal(elements.get('#example-template').open, true);
  assert.equal(postCount, 0);
  await elements.get('#method').change('reeval');
  await elements.get('#dataset').change('sar1');
  await elements.get('#models').change('internvl2b');
  await elements.get('#run-job').listeners.click();
  assert.equal(postCount, 1);
  assert.equal(elements.get('#job-progress').hidden, false);
  assert.match(elements.get('#job-status').textContent, /提交状态无法确认/);
  assert.equal(elements.get('#run-job').disabled, true);
  console.log('PASS: configuration and saved report UI without starting a job');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
