import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const [root, mode, scenarioArg, payloadPath] = process.argv.slice(2);
assert.ok(root && ['--inspect', '--export', '--layout-fix'].includes(mode));
const out = path.join(root, 'reports/experiments/exp009');
const previews = path.join(out, 'workbook-previews');
await fs.mkdir(previews, {recursive:true});
const scenarios = scenarioArg ? [scenarioArg] : ['1','2','3','4-2','4-3'];
const sha = async file => createHash('sha256').update(await fs.readFile(file)).digest('hex');
const near = (a,b,label) => assert.ok(typeof a === 'number' && Number.isFinite(a) && Math.abs(a-b)<1e-6, `${label}: ${a} vs ${b}`);
const iso = value => new Date(`${value}T00:00:00Z`);
const finalPayloadPath = path.join(out, 'final_payload.json');
const errorPattern = '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!';
const evidenceDir = path.join(out, 'evidence/workbooks');
await fs.mkdir(evidenceDir, {recursive:true});
const keepInspectDumpOutsideReport = async file => {
  const temporary = path.join(root,'.work/exp009-workbooks/inspect-dumps');
  await fs.mkdir(temporary,{recursive:true});
  try { await fs.rename(`${file}.inspect.ndjson`,path.join(temporary,`${path.basename(file)}.inspect.ndjson`)); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
};

for (const scenario of scenarios) {
  assert.ok(['1','2','3','4-2','4-3'].includes(scenario));
  if (mode === '--layout-fix') {
    assert.notEqual(scenario,'1');
    const file = path.join(out,`result${scenario}.xlsx`);
    const auditPath = path.join(evidenceDir,`result${scenario}-export.json`);
    const record = JSON.parse(await fs.readFile(auditPath,'utf8'));
    assert.equal(await sha(file),record.output_sha256);
    const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
    for (const name of ['计划购电量', ...(['3','4-3'].includes(scenario)?['调整购电量']:[])]) {
      saved.worksheets.getItem(name).getRange('EP1:EQ335').format.columnWidthPx = 160;
    }
    saved.recalculate();
    const output = await SpreadsheetFile.exportXlsx(saved);
    await output.save(file);
    await keepInspectDumpOutsideReport(file);
    await fs.copyFile(file,record.output_duplicate);
    record.layout_refinement = {reason:'Prevent the kWh unit closing bracket wrapping alone in the daily total header',
      changed_columns:'EP:EQ on purchase sheets only', previous_sha256:record.output_sha256, values_or_formulas_changed:false};
    record.output_sha256 = await sha(file);
    record.duplicate_sha256 = await sha(record.output_duplicate);
    await fs.writeFile(auditPath,JSON.stringify(record,null,2)+'\n');
    console.log(JSON.stringify({scenario,layout_refinement_complete:true,sha256:record.output_sha256}));
    continue;
  }
  const template = path.join(root, 'data/templates', `result${scenario}.xlsx`);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(template));
  const overview = await workbook.inspect({kind:'workbook,sheet,table', maxChars:3500,
    tableMaxRows:3, tableMaxCols:5, tableMaxCellChars:80});
  await fs.writeFile(path.join(previews, `template-${scenario}-structure.ndjson`), overview.ndjson);
  const sheets = scenario === '1' ? ['计划购电量','充放电量'] :
    ['计划购电量', ...(['3','4-3'].includes(scenario) ? ['调整购电量'] : []), '充放电量', '紧急购电量'];
  for (const name of sheets) assert.ok(workbook.worksheets.getItem(name));
  if (mode === '--inspect') {
    for (const name of sheets) {
      const range = scenario === '1' ? (name === '计划购电量' ? 'A1:B12' : 'A1:E7') :
        (name.includes('购电量') && name !== '紧急购电量' ? 'A1:H8' : name === '充放电量' ? 'A1:F9' : 'A1:C9');
      const rendered = await workbook.render({sheetName:name, range, scale:1.5, format:'png'});
      await fs.writeFile(path.join(previews, `template-${scenario}-${name}.png`), new Uint8Array(await rendered.arrayBuffer()));
    }
    console.log(JSON.stringify({scenario, template_sha256:await sha(template), sheets, read_only:true}));
    continue;
  }
  assert.ok(payloadPath && scenarios.length === 1, 'Export one explicit scenario payload per process');
  const payload = JSON.parse(await fs.readFile(finalPayloadPath, 'utf8'));
  const input = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
  assert.equal(payload.experiment_id, 'exp009');
  assert.equal(payload.mode, 'final');
  assert.equal(payload.protocol.candidate_count_per_question, 1);
  const result = scenario === '1' ? payload.q1 : payload.scenarios[scenario];
  assert.equal(result.verification.passed, true);
  assert.deepEqual(input, result.workbook_data);
  const archive = path.resolve(root, result.archive.path);
  assert.equal(await sha(archive), result.archive.sha256, 'Source archive must match verified final payload');
  assert.equal(input.target_filename, `result${scenario}.xlsx`);
  const file = path.join(out, input.target_filename);
  await assert.rejects(fs.access(file), undefined, 'Do not overwrite an existing exported result');
  if (scenario === '1') {
    assert.equal(result.final_trajectory_stage, 1);
    assert.equal(result.stages.length, 1);
    assert.equal(input.plan_rows.length, 144);
    assert.equal(input.battery_rows.length, 6);
    const plan = workbook.worksheets.getItem('计划购电量');
    const battery = workbook.worksheets.getItem('充放电量');
    plan.getRange('A2:B145').values = input.plan_rows;
    battery.getRange('A2:E7').values = input.battery_rows;
    plan.getRange('B1').values = [['购电量（kWh）']];
    battery.getRange('B1:C1').values = [['充电量（kWh）','放电量（kWh）']];
    battery.getRange('E1').values = [['储电量（kWh）']];
    plan.getRange('B2:B145').setNumberFormat('0.0000');
    battery.getRange('B2:C7').setNumberFormat('0.0000');
    battery.getRange('E2:E7').setNumberFormat('0.0000');
    plan.getRange('A1:B145').format.columnWidthPx = 155;
    battery.getRange('A1:E7').format.columnWidthPx = 155;
    plan.getRange('B2:B145').format.horizontalAlignment = 'right';
    battery.getRange('B2:C7').format.horizontalAlignment = 'right';
    battery.getRange('E2:E7').format.horizontalAlignment = 'right';
    plan.freezePanes.freezeRows(1);
  } else {
    assert.equal(input.dates.length, 334);
    assert.equal(input.dates[0], '2025-02-01');
    assert.equal(input.dates[333], '2025-12-31');
    assert.equal(input.interval_headers.length, 144);
    assert.equal(input.interval_headers[0], '00:00-00:10');
    assert.equal(input.interval_headers[143], '23:50-24:00');
    for (const key of ['original','final']) {
      assert.equal(input[key].length, 334);
      input[key].forEach(row => assert.equal(row.length, 144));
    }
    for (const [name,key] of [['计划购电量','original'], ...(['3','4-3'].includes(scenario) ? [['调整购电量','final']] : [])]) {
      const sheet = workbook.worksheets.getItem(name);
      // The supplied adjustment tab is intentionally empty. Extend the
      // neighboring purchase layout without adding or reordering worksheets.
      if (name === '调整购电量') sheet.getRange('A1:EQ335').copyFrom(workbook.worksheets.getItem('计划购电量').getRange('A1:EQ335'), 'all');
      sheet.getRange('A1:EQ1').values = [['日期\\时间', ...input.interval_headers, '全天购电量（kWh）', '全天总费用（元）']];
      sheet.getRange('A2:EQ335').values = input.dates.map((date,i) => [iso(date), ...input[key][i], null, input.fees[i]]);
      sheet.getRange('EP2:EP335').formulas = input.dates.map((_,i) => [`=SUM(B${i+2}:EO${i+2})`]);
      sheet.getRange('A2:A335').setNumberFormat('yyyy/mm/dd');
      sheet.getRange('B2:EP335').setNumberFormat('0.0000');
      sheet.getRange('EQ2:EQ335').setNumberFormat('0.00');
      sheet.getRange('B2:EQ335').format.horizontalAlignment = 'right';
      sheet.getRange('A2:A335').format.horizontalAlignment = 'center';
      sheet.getRange('A1:EQ1').format.wrapText = true;
      sheet.getRange('A1:EQ1').format.rowHeight = 34;
      sheet.getRange('A1:A335').format.columnWidthPx = 100;
      sheet.getRange('B1:EQ335').format.columnWidthPx = 122;
      sheet.getRange('EP1:EQ335').format.columnWidthPx = 160;
      sheet.freezePanes.freezeRows(1); sheet.freezePanes.freezeColumns(1);
    }
    const battery = workbook.worksheets.getItem('充放电量');
    assert.equal(input.battery.length, 2004);
    for (let day=1; day<334; day++) battery.getRange(`A${2+day*6}:F${7+day*6}`).copyFrom(battery.getRange('A2:F7'), 'all');
    battery.getRange('A2:F2005').clear({applyTo:'contents'});
    battery.getRange('A2:F2005').values = input.battery.map(row => [row[0] ? iso(row[0]) : null, ...row.slice(1)]);
    battery.getRange('C1:D1').values = [['充电量（kWh）','放电量（kWh）']];
    battery.getRange('F1').values = [['储电量（kWh）']];
    battery.getRange('A2:A2005').setNumberFormat('yyyy/mm/dd');
    battery.getRange('C2:D2005').setNumberFormat('0.0000');
    battery.getRange('F2:F2005').setNumberFormat('0.0000');
    battery.getRange('A2:B2005').format.horizontalAlignment = 'center';
    battery.getRange('C2:D2005').format.horizontalAlignment = 'right';
    battery.getRange('E2:E2005').format.horizontalAlignment = 'center';
    battery.getRange('F2:F2005').format.horizontalAlignment = 'right';
    for (const [col,width] of [['A',100],['B',135],['C',130],['D',130],['E',100],['F',130]]) battery.getRange(`${col}1:${col}2005`).format.columnWidthPx = width;
    battery.freezePanes.freezeRows(1); battery.freezePanes.freezeColumns(1);
    const emergency = workbook.worksheets.getItem('紧急购电量'), last = input.emergency.length+1;
    for (let row=3; row<=last; row++) emergency.getRange(`A${row}:C${row}`).copyFrom(emergency.getRange('A2:C2'), 'all');
    emergency.getRange(`A2:C${Math.max(last,11)}`).clear({applyTo:'contents'});
    emergency.getRange(`A2:C${last}`).values = input.emergency.map(row => [row[0] ? iso(row[0]) : null, ...row.slice(1)]);
    emergency.getRange('C1').values = [['购电量（kWh）']];
    emergency.getRange(`A2:A${last}`).setNumberFormat('yyyy/mm/dd');
    emergency.getRange(`C2:C${last}`).setNumberFormat('0.0000');
    emergency.getRange(`A2:B${last}`).format.horizontalAlignment = 'center';
    emergency.getRange(`C2:C${last}`).format.horizontalAlignment = 'right';
    for (const [col,width] of [['A',100],['B',155],['C',140]]) emergency.getRange(`${col}1:${col}${last}`).format.columnWidthPx = width;
    emergency.freezePanes.freezeRows(1);
    near(input.fees.reduce((a,b)=>a+b,0), result.fees.total_cost_yuan, 'Annual settled bill counted once');
  }
  workbook.recalculate();
  const errorScan = await workbook.inspect({kind:'match', searchTerm:errorPattern,
    options:{useRegex:true,maxResults:50}, maxChars:3000, summary:'exp009 formula error scan'});
  await fs.writeFile(path.join(evidenceDir, `result${scenario}-export-formula-scan.ndjson`), errorScan.ndjson);
  for (const name of sheets) {
    const cells = workbook.worksheets.getItem(name).getUsedRange(true).values;
    cells.forEach(row => row.forEach(value => assert.ok(typeof value !== 'string' || !/^#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!|SPILL!|CALC!)/.test(value))));
    const range = scenario === '1' ? (name === '计划购电量' ? 'A1:B12' : 'A1:E7') :
      (name.includes('购电量') && name !== '紧急购电量' ? 'A1:H8' : name === '充放电量' ? 'A1:F9' : 'A1:C9');
    const rendered = await workbook.render({sheetName:name, range, scale:1.5, format:'png'});
    await fs.writeFile(path.join(previews, `authored-${scenario}-${name}.png`), new Uint8Array(await rendered.arrayBuffer()));
  }
  const exportFile = await SpreadsheetFile.exportXlsx(workbook);
  await exportFile.save(file);
  await keepInspectDumpOutsideReport(file);
  const duplicateDir = path.join(root, 'outputs/exp009-cost-only-control');
  await fs.mkdir(duplicateDir, {recursive:true});
  await fs.copyFile(file, path.join(duplicateDir,input.target_filename));
  const record = {scenario, output:file, output_sha256:await sha(file), source_archive:result.archive,
    payload_sha256:await sha(payloadPath), final_payload_sha256:await sha(finalPayloadPath),
    template_sha256:await sha(template), template_sheets:sheets,
    output_duplicate:path.join(duplicateDir,input.target_filename),
    duplicate_sha256:await sha(path.join(duplicateDir,input.target_filename)),
    units:'purchase, charge, discharge and SOC in kWh; settled bill in yuan',
    fee_semantics:'Each daily total bill repeats on original and final purchase sheets; count it only once.',
    q1_stage:scenario === '1' ? 1 : null, prior_exp008_goal_gate_applied:false,
    export_complete:true, independent_cell_and_visual_QA:'pending'};
  await fs.writeFile(path.join(evidenceDir, `result${scenario}-export.json`), JSON.stringify(record,null,2)+'\n');
  console.log(JSON.stringify({scenario, exported:true, output:file, sha256:record.output_sha256}));
}
