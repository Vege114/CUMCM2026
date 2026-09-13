import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const [root,payloadFile] = process.argv.slice(2);
assert.ok(root && payloadFile, 'Usage: verify_final_workbooks.mjs ROOT FINAL_PAYLOAD_JSON');
const payload=JSON.parse(await fs.readFile(payloadFile,'utf8'));
assert.equal(payload.mode,'final'); assert.equal(payload.missing_items.length,0);
const out=path.join(root,'reports/experiments/exp008');
const auditDir=path.join(out,'evidence/workbooks');
await fs.mkdir(auditDir,{recursive:true});
const sha=async file=>createHash('sha256').update(await fs.readFile(file)).digest('hex');
const near=(value,expected,label)=>assert.ok(typeof value==='number' && Number.isFinite(value) && Math.abs(value-expected)<1e-6, `${label}: ${value} vs ${expected}`);
const serial=date=>(Date.parse(date+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000;
const dateValue=value=>value instanceof Date ? (value.getTime()-Date.UTC(1899,11,30))/86400000 : value;
const col=n=>{let r='';while(n){n--;r=String.fromCharCode(65+n%26)+r;n=Math.floor(n/26);}return r;};
const err=/^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/;
let scalarCount=0;
const compare=(actual,expected,label)=>{
  assert.equal(actual.length,expected.length,`${label} rows`);
  actual.forEach((row,i)=>{assert.equal(row.length,expected[i].length,`${label} columns`);row.forEach((v,j)=>{
    scalarCount++;
    if(typeof expected[i][j]==='number') near(v,expected[i][j],`${label} ${i},${j}`);
    else assert.equal(v??null,expected[i][j],`${label} ${i},${j}`);
  });});
};
const summary=[];
for(const scenario of ['1','2','3','4-2','4-3']) {
  const final=path.join(out,`result${scenario}.xlsx`);
  const source=scenario==='1'?path.join(out,'workbook-previews/q1/result1.xlsx'):path.join(root,'data/results/exp008',`result${scenario}.xlsx`);
  await fs.copyFile(source,final);
  if(scenario==='1') await fs.copyFile(final,path.join(root,'data/results/exp008/result1.xlsx'));
  const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(final));
  const before=scalarCount;
  let recalcChecks=0, formulaCount=0;
  const sheetRecords=[];
  if(scenario==='1') {
    const data=payload.q1.workbook_data;
    compare(workbook.worksheets.getItem('计划购电量').getRange('A2:B145').values,data.plan_rows,'Q1 purchase');
    compare(workbook.worksheets.getItem('充放电量').getRange('A2:E7').values,data.battery_rows,'Q1 battery');
    assert.equal(await sha(payload.q1.archive.path),payload.q1.archive.sha256);
    for(const name of ['计划购电量','充放电量'])sheetRecords.push({name,all_data_reimported:true});
  } else {
    const sc=payload.scenarios[scenario], data=sc.workbook_data;
    assert.equal(await sha(sc.archive.path),sc.archive.sha256);
    assert.equal(data.dates.length,334); assert.equal(data.dates[0],'2025-02-01');assert.equal(data.dates[333],'2025-12-31');
    for(const [name,key] of [['计划购电量','original'],...(['3','4-3'].includes(scenario)?[['调整购电量','final']]:[])]) {
      const sheet=workbook.worksheets.getItem(name);
      compare(sheet.getRange('B1:EO1').values,[data.interval_headers],`${scenario} headers`);
      const actual=sheet.getRange('A2:EQ335').values;
      actual.forEach(row=>row[0]=dateValue(row[0]));
      const expected=data.dates.map((date,i)=>[serial(date),...data[key][i],data[key][i].reduce((s,v)=>s+v,0),data.fees[i]]);
      compare(actual,expected,`${scenario} ${name}`);
      const formulas=sheet.getRange('EP2:EP335').formulas;
      compare(formulas,data.dates.map((_,i)=>[`=SUM(B${i+2}:EO${i+2})`]),`${scenario} ${name} sum formulas`);
      formulaCount+=334;
      for(const row of [2,169,335]) {
        const input=sheet.getRange(`B${row}`),total=sheet.getRange(`EP${row}`);
        const prior=input.values[0][0],sum=total.values[0][0];
        input.values=[[prior+.5]];workbook.recalculate();near(total.values[0][0],sum+.5,'SUM recalculation');
        input.values=[[prior]];workbook.recalculate();near(total.values[0][0],sum,'Restored SUM');recalcChecks++;
      }
      sheetRecords.push({name,rows:334,ten_minute_slots:144,all_data_reimported:true,fee_column:'EQ',fee_semantics:'daily total settled bill, repeated on both purchase sheets; count once'});
    }
    for(const [name,rows,width] of [['充放电量',data.battery,6],['紧急购电量',data.emergency,3]]) {
      const range=workbook.worksheets.getItem(name).getRange(`A2:${col(width)}${rows.length+1}`);
      const actual=range.values;actual.forEach(row=>row[0]=row[0]==null?null:dateValue(row[0]));
      const expected=rows.map(row=>[row[0]?serial(row[0]):null,...row.slice(1)]);
      compare(actual,expected,`${scenario} ${name}`);
      sheetRecords.push({name,rows:rows.length,all_data_reimported:true});
    }
    near(data.fees.reduce((s,v)=>s+v,0),sc.fees.total_cost_yuan,'Annual bill once');
    assert.equal(data.battery.length,2004);
  }
  for(const {name} of sheetRecords) {
    const values=workbook.worksheets.getItem(name).getUsedRange(true).values;
    values.forEach(row=>row.forEach(v=>assert.ok(typeof v!=='string'||!err.test(v),`Formula error ${v}`)));
  }
  workbook.recalculate();
  const savedPreviews=[];
  for(const {name} of sheetRecords) {
    const range=scenario==='1'?(name==='计划购电量'?'A1:B12':'A1:E7'):
      (name==='计划购电量'||name==='调整购电量'?'A1:H8':name==='充放电量'?'A1:F9':'A1:C9');
    // Render every sheet from the saved file, independent of in-memory tests.
    const renderBook=await SpreadsheetFile.importXlsx(await FileBlob.load(final));
    const preview=await renderBook.render({sheetName:name,range,scale:1.5,format:'png'});
    const file=path.join(out,'workbook-previews',`saved-${scenario}-${name}.png`);
    await fs.writeFile(file,new Uint8Array(await preview.arrayBuffer()));
    savedPreviews.push({sheet:name,range,path:file,sha256:await sha(file)});
  }
  const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:3000,summary:'Final saved workbook formula error scan'});
  await fs.writeFile(path.join(auditDir,`result${scenario}-formula-scan.ndjson`),errors.ndjson);
  const record={scenario,path:final,sha256:await sha(final),source_copy_sha256:await sha(source),source_archive:scenario==='1'?payload.q1.archive:payload.scenarios[scenario].archive,
    all_exported_cell_values_compared:true,scalar_comparisons:scalarCount-before,sheets:sheetRecords,sum_formulas_checked:formulaCount,
    input_change_recalculation_checks:recalcChecks,unexpected_formula_errors:0,
    engine:'Artifact Tool 2.8.58+ bundled runtime; Excel desktop not launched',units:'kWh for purchase, battery charge/discharge and SOC; yuan for settled bill',
    formula_scan_file:`result${scenario}-formula-scan.ndjson`,saved_file_previews:savedPreviews};
  summary.push(record);console.log(JSON.stringify({scenario,passed:true,scalar_comparisons:record.scalar_comparisons,recalcChecks}));
}
await fs.writeFile(path.join(auditDir,'saved_workbook_audit.json'),JSON.stringify({passed:true,payload_sha256:await sha(payloadFile),files:summary,total_scalar_comparisons:scalarCount,visual_QA_status:'pending',no_optimizer_rerun:true},null,2)+'\n');
