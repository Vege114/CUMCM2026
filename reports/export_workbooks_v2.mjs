import fs from "node:fs/promises";
import path from "node:path";
import {FileBlob, SpreadsheetFile} from "@oai/artifact-tool";

const [root, runId="exp002", mode="export", onlyScenario] = process.argv.slice(2);
if(!root) throw new Error("Usage: node export_workbooks_v2.mjs ROOT RUN_ID [inspect|export]");
const report=path.join(root,"reports/experiments",runId);
await fs.mkdir(path.join(report,"workbook-previews"),{recursive:true});
const column=n=>{let x="";while(n){n--;x=String.fromCharCode(65+n%26)+x;n=Math.floor(n/26);}return x;};
const clock=slot=>`${String(Math.floor(slot/6)).padStart(2,"0")}:${String((slot%6)*10).padStart(2,"0")}`;
const isoDate=value=>new Date(`${value}T00:00:00Z`);
for(const scenario of onlyScenario?[onlyScenario]:["2","3","4-2","4-3"]){
  const fileName=`result${scenario}.xlsx`;
  const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,"data/templates",fileName)));
  const sheets=["计划购电量",...(["3","4-3"].includes(scenario)?["调整购电量"]:[]),"充放电量","紧急购电量"];
  if(mode==="inspect"){
    console.log(fileName,await workbook.inspect({kind:"sheet",include:"id,name"}));
    for(const name of sheets){
      const preview=await workbook.render({sheetName:name,range:name.includes("购电量")&&name!=="紧急购电量"?"A1:H8":"A1:F9",scale:1.5,format:"png"});
      await fs.writeFile(path.join(report,"workbook-previews",`template-${scenario}-${name}.png`),new Uint8Array(await preview.arrayBuffer()));
    }
    continue;
  }
  const input=JSON.parse(await fs.readFile(path.join(root,".work",runId,`workbook-${scenario}.json`),"utf8"));
  for(const [name,key] of [["计划购电量","original"],["调整购电量","final"]]){
    if(name==="调整购电量"&&!["3","4-3"].includes(scenario))continue;
    const sheet=workbook.worksheets.getItem(name);
    sheet.getRange(`B1:${column(145)}1`).values=[Array.from({length:144},(_,slot)=>`${clock(slot)}-${clock(slot+1)}`)];
    const rows=input.dates.map((date,i)=>[isoDate(date),...input[key][i],null,input.fees[i]]);
    sheet.getRange(`A2:${column(147)}335`).values=rows;
    sheet.getRange(`${column(146)}2:${column(146)}335`).formulas=input.dates.map((_,i)=>[`=SUM(B${i+2}:${column(145)}${i+2})`]);
    sheet.getRange("A2:A335").setNumberFormat("yyyy/mm/dd");
    sheet.getRange(`B2:${column(146)}335`).setNumberFormat("0.0000");
    sheet.getRange(`B2:${column(147)}335`).format.horizontalAlignment="right";
    sheet.getRange("A2:A335").format.horizontalAlignment="center";
    sheet.getRange(`${column(147)}2:${column(147)}335`).setNumberFormat("0.00");
    sheet.getRange(`A1:${column(147)}1`).format.wrapText=true;
    sheet.getRange(`A1:${column(147)}1`).format.rowHeight=34;
    sheet.getRange("A1:A335").format.columnWidthPx=100;
    sheet.getRange(`B1:${column(147)}335`).format.columnWidthPx=122;
    sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(1);
  }
  const battery=workbook.worksheets.getItem("充放电量");
  for(let day=1;day<334;day++)battery.getRange(`A${2+day*6}:F${7+day*6}`).copyFrom(battery.getRange("A2:F7"),"all");
  battery.getRange("A2:F2005").clear({applyTo:"contents"});
  battery.getRange("A2:F2005").values=input.battery.map(row=>[row[0]?isoDate(row[0]):null,...row.slice(1)]);
  battery.getRange("A2:A2005").setNumberFormat("yyyy/mm/dd");
  battery.getRange("C2:D2005").setNumberFormat("0.0000");
  battery.getRange("F2:F2005").setNumberFormat("0.0000");
  battery.getRange("A2:B2005").format.horizontalAlignment="center";
  battery.getRange("C2:D2005").format.horizontalAlignment="right";
  battery.getRange("E2:E2005").format.horizontalAlignment="center";
  battery.getRange("F2:F2005").format.horizontalAlignment="right";
  for(const [col,width] of [["A",100],["B",135],["C",120],["D",120],["E",100],["F",120]])battery.getRange(`${col}1:${col}2005`).format.columnWidthPx=width;
  battery.freezePanes.freezeRows(1);battery.freezePanes.freezeColumns(1);
  const emergency=workbook.worksheets.getItem("紧急购电量"),last=input.emergency.length+1;
  for(let row=3;row<=last;row++)emergency.getRange(`A${row}:C${row}`).copyFrom(emergency.getRange("A2:C2"),"all");
  emergency.getRange(`A2:C${Math.max(last,11)}`).clear({applyTo:"contents"});
  emergency.getRange(`A2:C${last}`).values=input.emergency.map(row=>[row[0]?isoDate(row[0]):null,...row.slice(1)]);
  emergency.getRange(`A2:A${last}`).setNumberFormat("yyyy/mm/dd");
  emergency.getRange(`C2:C${last}`).setNumberFormat("0.0000");
  emergency.getRange(`A2:B${last}`).format.horizontalAlignment="center";
  emergency.getRange(`C2:C${last}`).format.horizontalAlignment="right";
  for(const [col,width] of [["A",100],["B",155],["C",140]])emergency.getRange(`${col}1:${col}${last}`).format.columnWidthPx=width;
  emergency.freezePanes.freezeRows(1);
  workbook.recalculate();
  console.log(fileName,await workbook.inspect({kind:"table",range:`计划购电量!${column(144)}1:${column(147)}4`,include:"values,formulas",maxChars:1800}));
  for(const name of sheets){
    const preview=await workbook.render({sheetName:name,range:name.includes("购电量")&&name!=="紧急购电量"?"A1:H8":"A1:F9",scale:1.5,format:"png"});
    await fs.writeFile(path.join(report,"workbook-previews",`${scenario}-${name}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  const totals=await workbook.render({sheetName:"计划购电量",range:`${column(144)}1:${column(147)}8`,scale:1.5,format:"png"});
  await fs.writeFile(path.join(report,"workbook-previews",`${scenario}-totals.png`),new Uint8Array(await totals.arrayBuffer()));
  const output=await SpreadsheetFile.exportXlsx(workbook);
  await output.save(path.join(root,"data/results",runId,fileName));
  console.log("EXPORTED",fileName);
}
