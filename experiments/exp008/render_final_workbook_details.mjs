import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const [root]=process.argv.slice(2);
const dir=path.join(root,'reports/experiments/exp008');
const ledger=[];
for(const scenario of ['2','3','4-2','4-3']){
  const file=path.join(dir,`result${scenario}.xlsx`);
  const views=[['计划购电量','EN1:EQ8','totals'],['计划购电量','A329:H335','last-days']];
  if(['3','4-3'].includes(scenario))views.push(['调整购电量','AX1:BE8','update-eight-hours']);
  for(const [sheetName,range,name] of views){
    const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(file));
    const image=await wb.render({sheetName,range,scale:1.5,format:'png'});
    const output=path.join(dir,'workbook-previews',`saved-${scenario}-${name}.png`);
    await fs.writeFile(output,new Uint8Array(await image.arrayBuffer()));
    ledger.push({scenario,sheet:sheetName,range,path:output,sha256:createHash('sha256').update(await fs.readFile(output)).digest('hex')});
  }
}
await fs.writeFile(path.join(dir,'evidence/workbooks/detail_previews.json'),JSON.stringify(ledger,null,2)+'\n');
console.log(JSON.stringify({saved_file_detail_previews:ledger.length}));
