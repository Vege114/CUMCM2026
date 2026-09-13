import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const [root, mode] = process.argv.slice(2);
assert.ok(root);
const out = path.join(root,'reports/experiments/exp009');
const evidence = path.join(out,'evidence/workbooks');
const previewDir = path.join(out,'workbook-previews');
const payloadPath = path.join(out,'final_payload.json');
const payload = JSON.parse(await fs.readFile(payloadPath,'utf8'));
assert.equal(payload.experiment_id,'exp009'); assert.equal(payload.mode,'final');
const sha = async file => createHash('sha256').update(await fs.readFile(file)).digest('hex');
const near = (a,b,label) => assert.ok(typeof a === 'number' && Number.isFinite(a) && Math.abs(a-b)<1e-6, `${label}: ${a} vs ${b}`);
const serial = value => (Date.parse(value+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000;
const dateValue = value => value instanceof Date ? (value.getTime()-Date.UTC(1899,11,30))/86400000 : value;
let cells = 0;
const compare = (a,b,label) => {
  assert.equal(a.length,b.length,label+' rows');
  a.forEach((row,i) => {assert.equal(row.length,b[i].length,label+' cols'); row.forEach((v,j) => {
    cells++;
    if (typeof b[i][j] === 'number') near(v,b[i][j],`${label} ${i},${j}`);
    else assert.equal(v??null,b[i][j],`${label} ${i},${j}`);
  });});
};
const records = [];
const priorAudit = mode === '--refresh-totals' ? JSON.parse(await fs.readFile(path.join(evidence,'saved_workbook_audit.json'),'utf8')) : null;
for (const scenario of ['1','2','3','4-2','4-3']) {
  const file = path.join(out,`result${scenario}.xlsx`), beforeHash = await sha(file);
  if (priorAudit && scenario === '1') {
    const prior = priorAudit.files.find(row=>row.scenario === scenario);
    assert.equal(prior.sha256,beforeHash);
    records.push(prior); cells += prior.scalar_comparisons;
    continue;
  }
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const result = scenario === '1' ? payload.q1 : payload.scenarios[scenario];
  assert.equal(result.verification.passed,true);
  assert.equal(await sha(path.resolve(root,result.archive.path)),result.archive.sha256);
  const data = result.workbook_data, views = [], startCount = cells;
  let recalcChecks = 0, formulas = 0;
  if (scenario === '1') {
    assert.equal(result.final_trajectory_stage,1); assert.equal(result.stages.length,1);
    compare(workbook.worksheets.getItem('计划购电量').getRange('A2:B145').values,data.plan_rows,'Q1 plan');
    compare(workbook.worksheets.getItem('充放电量').getRange('A2:E7').values,data.battery_rows,'Q1 battery');
    views.push(['计划购电量','A1:B12','plan-first'],['计划购电量','A68:B78','plan-middle'],
               ['计划购电量','A135:B145','plan-last'],['充放电量','A1:E7','battery']);
  } else {
    for (const [name,key] of [['计划购电量','original'],...(['3','4-3'].includes(scenario)?[['调整购电量','final']]:[])]) {
      const sheet = workbook.worksheets.getItem(name);
      compare(sheet.getRange('B1:EO1').values,[data.interval_headers],`${scenario} ${name} headers`);
      const actual = sheet.getRange('A2:EQ335').values;
      actual.forEach(row=>row[0]=dateValue(row[0]));
      compare(actual,data.dates.map((date,i)=>[serial(date),...data[key][i],data[key][i].reduce((a,b)=>a+b,0),data.fees[i]]),`${scenario} ${name}`);
      compare(sheet.getRange('EP2:EP335').formulas,data.dates.map((_,i)=>[`=SUM(B${i+2}:EO${i+2})`]),`${scenario} ${name} formulas`);
      formulas += 334;
      for (const row of [2,169,335]) {
        const input = sheet.getRange(`B${row}`), output = sheet.getRange(`EP${row}`);
        const v = input.values[0][0], total = output.values[0][0];
        input.values=[[v+.5]]; workbook.recalculate(); near(output.values[0][0],total+.5,'live sum update');
        input.values=[[v]]; workbook.recalculate(); near(output.values[0][0],total,'restored sum');
        recalcChecks++;
      }
      views.push([name,'A1:H8',name],[name,'EN1:EQ8',`${name}-totals`],[name,'A328:H335',`${name}-last-days`]);
      if (name === '调整购电量') views.push([name,'AH1:AQ8','six-hour-update']);
    }
    for (const [name,rows,column] of [['充放电量',data.battery,'F'],['紧急购电量',data.emergency,'C']]) {
      const actual=workbook.worksheets.getItem(name).getRange(`A2:${column}${rows.length+1}`).values;
      actual.forEach(row=>row[0]=row[0]==null?null:dateValue(row[0]));
      compare(actual,rows.map(row=>[row[0]?serial(row[0]):null,...row.slice(1)]),`${scenario} ${name}`);
      views.push([name,`A1:${column}9`,name],[name,`A${rows.length-7}:${column}${rows.length+1}`,`${name}-last`]);
    }
    near(data.fees.reduce((a,b)=>a+b,0),result.fees.total_cost_yuan,'bill counted once');
  }
  workbook.recalculate();
  const sheets = [...new Set(views.map(row=>row[0]))];
  for (const name of sheets) workbook.worksheets.getItem(name).getUsedRange(true).values.forEach(row=>row.forEach(value=>{
    assert.ok(typeof value!=='string'||!/^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/.test(value),`formula error ${value}`);
  }));
  const scan = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
    options:{useRegex:true,maxResults:50},maxChars:3000,summary:'saved exp009 formula scan'});
  await fs.writeFile(path.join(evidence,`result${scenario}-saved-formula-scan.ndjson`),scan.ndjson);
  const previewRecords=priorAudit ? priorAudit.files.find(row=>row.scenario===scenario).saved_file_previews.filter(row=>!row.path.includes('-totals.png')) : [];
  for (const [sheetName,range,label] of views) {
    if (priorAudit && !label.endsWith('-totals')) continue;
    const preview=await workbook.render({sheetName,range,scale:1.5,format:'png'});
    const previewPath=path.join(previewDir,`saved-${scenario}-${label}.png`);
    await fs.writeFile(previewPath,new Uint8Array(await preview.arrayBuffer()));
    previewRecords.push({sheetName,range,path:previewPath,sha256:await sha(previewPath)});
  }
  assert.equal(await sha(file),beforeHash,'verification must not mutate saved output');
  records.push({scenario,path:file,sha256:beforeHash,passed:true,source_archive:result.archive,
    scalar_comparisons:cells-startCount,all_exported_data_compared:true,
    sum_formulas_checked:formulas,input_change_restore_recalculation_checks:recalcChecks,
    unexpected_formula_errors:0,prior_exp008_goal_gate_applied:false,saved_file_previews:previewRecords,
    engine:'Bundled @oai/artifact-tool; Excel desktop was not launched'});
  console.log(JSON.stringify({scenario,passed:true,scalar_comparisons:cells-startCount,recalcChecks}));
}
await fs.writeFile(path.join(evidence,'saved_workbook_audit.json'),JSON.stringify({passed:true,
  payload_sha256:await sha(payloadPath),files:records,total_scalar_comparisons:cells,
  source_archive_independent_reconstruction:'independent_cell_audit.json',visual_QA_status:'see visual_review.json'},null,2)+'\n');
