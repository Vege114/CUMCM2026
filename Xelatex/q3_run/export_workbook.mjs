// 第三问附件工作簿导出：Node.js + @oai/artifact-tool。
// 数值由独立核验后的workbook_payload.json输入；不训练、不重新优化。
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const path=(relative)=>fileURLToPath(new URL(relative,import.meta.url));
const payload=JSON.parse(await fs.readFile(path('paper/workbook_payload.json'),'utf8'));
const asDate=(value)=>value ? new Date(value+'T00:00:00Z') : null;
const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(path('inputs/result3_template.xlsx')));
const audit=workbook.worksheets.add('计费核验');
audit.getRange('A1:F1').values=[['日期','原计划费/元','上调费/元','下调退款/元','紧急费/元','总费/元']];
audit.getRange('A2:F335').values=payload.dates.map((date,i)=>[asDate(date),...payload.components[i],null]);
audit.getRange('A2:A335').setNumberFormat('yyyy-mm-dd');
audit.getRange('F2:F335').formulas=payload.dates.map((_,i)=>[`=SUM(B${i+2}:E${i+2})`]);
audit.getRange('A337:E337').values=[['口径：原计划+1.5倍上调−0.5倍下调+5倍紧急；最终净调整只结算一次。',null,null,null,null]];
audit.getRange('A337:F337').merge();
audit.getRange('A338:F338').merge();
audit.getRange('A338').values=[['来源：严格物理情景MILP完整年度实际轨迹；日期范围2025-02-01至12-31。']];
for (const [name,key] of [['计划购电量','original'],['调整购电量','final']]) {
  const sheet=workbook.worksheets.getItem(name);
  sheet.getUsedRange().clear({applyTo:'contents'}); // 保留附件样式，修正十分钟标签。
  sheet.getRange('A1:EQ1').values=[payload.headers];
  sheet.getRange('A2:EQ335').values=payload.dates.map((date,i)=>[asDate(date),...payload[key][i],null,null]);
  sheet.getRange('A2:A335').setNumberFormat('yyyy-mm-dd');
  sheet.getRange('EP2:EP335').formulas=payload.dates.map((_,i)=>[`=SUM(B${i+2}:EO${i+2})`]);
  sheet.getRange('EQ2:EQ335').formulas=payload.dates.map((_,i)=>[`='计费核验'!${key==='original'?'B':'F'}${i+2}`]);
  sheet.getRange('B2:EQ335').setNumberFormat('0.000000');
  sheet.getRange('A1:EQ335').format.columnWidth=15;
  sheet.getRange('A1:EQ1').format.wrapText=true;
  sheet.getRange('A1:EQ1').format.rowHeight=32;
  sheet.freezePanes.freezeRows(1);
}
const storage=workbook.worksheets.getItem('充放电量');
storage.getUsedRange().unmerge(); storage.getUsedRange().clear({applyTo:'contents'});
storage.getRange('A1:F1').values=[['日期','时间段','充电量/kWh','放电量/kWh','时刻','储电量/kWh']];
storage.getRangeByIndexes(1,0,payload.storage.length,6).values=payload.storage.map(row=>[asDate(row[0]),...row.slice(1)]);
storage.getRangeByIndexes(1,0,payload.storage.length,1).setNumberFormat('yyyy-mm-dd');
storage.getRangeByIndexes(1,2,payload.storage.length,2).setNumberFormat('0.000000');
storage.getRangeByIndexes(1,5,payload.storage.length,1).setNumberFormat('0.000000');
const emergency=workbook.worksheets.getItem('紧急购电量');
emergency.getUsedRange().unmerge(); emergency.getUsedRange().clear({applyTo:'contents'});
emergency.getRange('A1:C1').values=[['日期','紧急购电时间段','购电量/kWh']];
if(payload.emergency.length) {
  emergency.getRangeByIndexes(1,0,payload.emergency.length,3).values=payload.emergency.map(row=>[asDate(row[0]),...row.slice(1)]);
  emergency.getRangeByIndexes(1,0,payload.emergency.length,1).setNumberFormat('yyyy-mm-dd');
  emergency.getRangeByIndexes(1,2,payload.emergency.length,1).setNumberFormat('0.000000');
}
for(const sheet of [storage,emergency,audit]) {
  sheet.getUsedRange().format.columnWidth=20;
  sheet.getRangeByIndexes(0,0,1,sheet===emergency?3:6).format={font:{bold:true},rowHeight:26};
  sheet.freezePanes.freezeRows(1);
}
audit.getRange('B2:F335').setNumberFormat('#,##0.00;[Red](#,##0.00)');
const preview=await workbook.render({sheetName:'计费核验',range:'A1:F9',scale:1,format:'png'});
await fs.writeFile(path('paper/workbook_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const result=await SpreadsheetFile.exportXlsx(workbook);
await result.save(path('result3.xlsx'));
const inspection=await workbook.inspect({kind:'region',sheetId:'计费核验',range:'A1:F5',maxChars:2500});
await fs.writeFile(path('paper/workbook_inspection.ndjson'),inspection.ndjson);
console.log('result3.xlsx exported');
